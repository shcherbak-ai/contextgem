# Extraction evaluation: baseline_2026-10-23

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_dpa | anthropic/claude-haiku-5-5 | default | 3 | 0.909 (0.902-0.923) | 21.3/22 (21-22) | 5.7/6 (5-6) | 0.860 (0.849-0.866) | 7 (7-8) | 86,260 (83,165-92,146) | 0 | 13,203 (12,720-14,054) | 58 (56-62) | 1 |
| complex_dpa | azure_ai/ms-decision-model | decision_default | 2 | 0.772 (0.758-0.785) | 5/6 | 5/6 | n/a | 157 | 109,228 (109,183-109,273) | 0 | 934 | 47 (46-47) | 0 |
| complex_dpa | openai/gpt-6-luna | concurrent | 3 | 0.953 (0.941-0.969) | 21.7/22 (21-22) | 6/6 | 0.901 (0.890-0.912) | 30 | 228,680 (228,475-228,939) | 134,889 (48,830-177,919) | 11,905 (10,834-12,473) | 33 (28-36) | 0 |
| complex_dpa | openai/gpt-6-luna | decision_default | 2 | 0.823 | 4/6 | 4/6 | n/a | 157 | 222,536 | 2,821 (0-5,642) | 0 | 39 (38-40) | 0 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 0.979 (0.978-0.981) | 21.7/22 (21-22) | 6/6 | 0.848 (0.833-0.867) | 7 | 51,649 (51,597-51,720) | 31,887 (9,569-43,046) | 7,472 (7,182-7,620) | 69 (66-72) | 0 |
| complex_dpa | openai/gpt-6.1-sol | default | 1 | 0.895 | 22/22 | 6/6 | 0.927 | 7 | 52,802 | 9,569 | 4,449 | 86 | 0 |
| complex_dpa | typesafe/jev-latest | decision_default | 2 | 0.777 (0.774-0.779) | 5/6 | 5/6 | n/a | 157 | 143,866 (143,802-143,931) | 0 | 16,545 | 29 (28-29) | 0 |
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 0.690 (0.664-0.712) | 19.7/26 (17-21) | 8.3/11 (7-9) | 0.630 (0.594-0.690) | 10 | 126,087 (123,143-128,988) | 0 | 17,460 (16,320-19,264) | 83 (77-93) | 0 |
| complex_mcu | azure_ai/ms-decision-model | decision_default | 2 | 0.564 (0.561-0.566) | 8.5/11 (8-9) | 8.5/11 (8-9) | n/a | 275 | 282,645 (282,516-282,774) | 0 | 1,903 | 82 (80-83) | 0 |
| complex_mcu | openai/gpt-6-luna | concurrent | 3 | 0.901 (0.867-0.919) | 25.0/26 (24-26) | 11/11 | 0.764 (0.755-0.773) | 34 | 337,843 (336,378-339,044) | 180,518 (46,478-257,752) | 26,849 (24,802-29,153) | 63 (56-73) | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 2 | 0.633 | 8/11 | 8/11 | n/a | 275 | 517,005 | 6,721 (0-13,442) | 0 | 76 (74-78) | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.923 (0.908-0.943) | 24.3/26 (23-26) | 10.7/11 (10-11) | 0.698 (0.667-0.728) | 10 | 83,269 (82,443-83,848) | 36,587 (13,670-48,191) | 14,551 (13,355-15,769) | 140 (125-155) | 0 |
| complex_mcu | openai/gpt-6.1-sol | default | 1 | 0.942 | 26/26 | 11/11 | 0.745 | 10 | 84,067 | 13,670 | 6,646 | 145 | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 2 | 0.647 (0.644-0.651) | 8/11 | 8/11 | n/a | 275 | 339,169 (339,054-339,284) | 0 | 33,516 | 48 (48-49) | 0 |
| complex_nda | anthropic/claude-haiku-5-5 | default | 3 | 0.951 (0.930-0.981) | 17.7/18 (17-18) | 6/6 | 0.829 (0.802-0.867) | 7 (6-8) | 80,827 (69,113-103,485) | 0 | 10,413 (9,105-12,697) | 48 (42-58) | 2 |
| complex_nda | azure_ai/ms-decision-model | decision_default | 2 | 0.771 (0.767-0.775) | 4/6 | 4/6 | n/a | 76 | 50,964 (50,933-50,996) | 0 | 312 | 23 (22-24) | 0 |
| complex_nda | openai/gpt-6-luna | concurrent | 3 | 0.975 (0.961-1.000) | 18/18 | 6/6 | 0.852 (0.845-0.858) | 25 | 173,067 (171,785-174,049) | 84,665 (34,175-115,319) | 9,335 (8,697-9,720) | 28 (25-32) | 0 |
| complex_nda | openai/gpt-6-luna | decision_default | 2 | 0.892 | 4/6 | 4/6 | n/a | 76 | 86,263 | 1,344 (0-2,689) | 0 | 19 (17-21) | 0 |
| complex_nda | openai/gpt-6-luna | default | 3 | 0.995 (0.992-1.000) | 18/18 | 6/6 | 0.867 (0.853-0.891) | 6 | 42,908 (42,829-42,986) | 24,904 (8,202-33,554) | 5,748 (5,327-6,214) | 54 (47-62) | 0 |
| complex_nda | openai/gpt-6.1-sol | default | 1 | 1.000 | 18/18 | 6/6 | 0.912 | 6 | 42,991 | 8,202 | 3,559 | 66 | 0 |
| complex_nda | typesafe/jev-latest | decision_default | 2 | 0.818 (0.803-0.833) | 3.5/6 (3-4) | 3.5/6 (3-4) | n/a | 76 | 64,931 (64,788-65,074) | 0 | 5,652 | 14 (14-14) | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 0.900 (0.856-0.940) | 13.7/16 (13-14) | 4/5 | 0.729 (0.713-0.738) | 7 | 69,887 (69,835-69,948) | 0 | 6,317 (5,557-7,041) | 30 (26-34) | 0 |
| complex_stability | azure_ai/ms-decision-model | decision_default | 2 | 0.372 (0.360-0.384) | 2/5 | 2/5 | n/a | 128 | 63,980 (63,936-64,023) | 0 | 511 | 38 (38-39) | 0 |
| complex_stability | openai/gpt-6-luna | concurrent | 3 | 0.959 (0.951-0.975) | 15.0/16 (14-16) | 4.7/5 (4-5) | 0.765 (0.711-0.804) | 22 | 132,818 (132,309-133,246) | 81,068 (37,113-108,690) | 6,880 (6,650-7,187) | 29 (26-31) | 0 |
| complex_stability | openai/gpt-6-luna | decision_default | 2 | 0.414 | 3/5 | 3/5 | n/a | 128 | 127,173 | 1,660 (0-3,321) | 0 | 32 (30-33) | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.942 (0.925-0.961) | 14.3/16 (14-15) | 4.3/5 (4-5) | 0.756 (0.733-0.784) | 7 | 44,850 (44,782-44,934) | 26,956 (9,569-37,015) | 4,775 (4,014-5,762) | 53 (43-64) | 0 |
| complex_stability | openai/gpt-6.1-sol | default | 1 | 0.984 | 16/16 | 5/5 | 0.763 | 7 | 45,008 | 9,569 | 2,974 | 55 | 0 |
| complex_stability | typesafe/jev-latest | decision_default | 2 | 0.380 (0.379-0.381) | 3/5 | 3/5 | n/a | 128 | 92,698 (92,695-92,700) | 0 | 9,241 | 23 (22-23) | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_dpa | anthropic/claude-haiku-5-5 | default | 3 | 3/6 (0.50) | 60/66 (0.91) | 24/24 (1.00) | 81 | 0.0017 |
| complex_dpa | openai/gpt-6-luna | concurrent | 3 | 4/6 (0.67) | 59/66 (0.89) | 24/24 (1.00) | 69 | 0.0015 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 4/6 (0.67) | 63/66 (0.95) | 24/24 (1.00) | 69 | 0.0014 |
| complex_dpa | openai/gpt-6.1-sol | default | 1 | 2/2 (1.00) | 21/22 (0.95) | 8/8 (1.00) | 27 | 0.0006 |
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 3/3 (1.00) | 57/68 (0.84) | 36/36 (1.00) | 68 | 0.0015 |
| complex_mcu | openai/gpt-6-luna | concurrent | 3 | 3/3 (1.00) | 69/78 (0.88) | 36/36 (1.00) | 79 | 0.0017 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 68/77 (0.88) | 36/36 (1.00) | 79 | 0.0017 |
| complex_mcu | openai/gpt-6.1-sol | default | 1 | 1/1 (1.00) | 25/26 (0.96) | 12/12 (1.00) | 27 | 0.0006 |
| complex_nda | anthropic/claude-haiku-5-5 | default | 3 | 7/9 (0.78) | 61/63 (0.97) | 12/12 (1.00) | 69 | 0.0015 |
| complex_nda | openai/gpt-6-luna | concurrent | 3 | 7/9 (0.78) | 62/63 (0.98) | 12/12 (1.00) | 68 | 0.0015 |
| complex_nda | openai/gpt-6-luna | default | 3 | 8/9 (0.89) | 63/63 (1.00) | 12/12 (1.00) | 71 | 0.0015 |
| complex_nda | openai/gpt-6.1-sol | default | 1 | 3/3 (1.00) | 21/21 (1.00) | 4/4 (1.00) | 22 | 0.0005 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 6/6 (1.00) | 44/50 (0.88) | 18/18 (1.00) | 51 | 0.0011 |
| complex_stability | openai/gpt-6-luna | concurrent | 3 | 6/6 (1.00) | 48/51 (0.94) | 18/18 (1.00) | 54 | 0.0012 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 46/51 (0.90) | 18/18 (1.00) | 54 | 0.0012 |
| complex_stability | openai/gpt-6.1-sol | default | 1 | 2/2 (1.00) | 17/17 (1.00) | 6/6 (1.00) | 18 | 0.0004 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 64 | 4,362 | 1.4414 |
| Judge | 41 | 982 | 0.0211 |
| Decision probabilities | 12 | 1,854 | 0.1338 |
| Total | | | 1.5963 |

## Concepts missed

- **complex_dpa, anthropic/claude-haiku-5-5, default** (3 runs): Held ORD records deletion deadline (DateConcept): 1x; Liability cap covers confidentiality breaches (BooleanConcept): 1x
- **complex_dpa, azure_ai/ms-decision-model, decision_default** (2 runs): Applicable initial notification regime (LabelConcept): 2x
- **complex_dpa, openai/gpt-6-luna, concurrent** (3 runs): Held ORD records deletion deadline (DateConcept): 1x
- **complex_dpa, openai/gpt-6-luna, decision_default** (2 runs): Meridian Mail approved on 1 February 2027 (BooleanConcept): 2x; Permitted uses of held ORD records on 1 September 2027 (LabelConcept): 2x
- **complex_dpa, openai/gpt-6-luna, default** (3 runs): Distinct data locations count as of 1 February 2027 (NumericalConcept): 1x
- **complex_dpa, typesafe/jev-latest, decision_default** (2 runs): Meridian Mail approved on 1 November 2026 (BooleanConcept): 1x; Authorised data locations on 15 May 2027 (LabelConcept): 1x
- **complex_mcu, anthropic/claude-haiku-5-5, default** (3 runs): DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 3x; DR-114 junction temperature in °C (NumericalConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 3x; Errata applicable to the DR-115 boards (LabelConcept): 3x; HCLK maximum for 2 wait states in VOS1 in MHz (NumericalConcept): 2x; DR-114 required flash wait states (NumericalConcept): 2x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; DR-115 maximum PLL1 VCO frequency in MHz (NumericalConcept): 1x; DR-115 PLL1 configuration within its limits (BooleanConcept): 1x
- **complex_mcu, azure_ai/ms-decision-model, decision_default** (2 runs): DR-114 PLL1 configuration within specification (BooleanConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 2x; Errata applicable to the DR-114 production boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, concurrent** (3 runs): DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 2x; Last-shipment date of revision B (DateConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (2 runs): Errata applicable to the DR-114 production boards (LabelConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 2x; PO-5521 valid last-time-buy order (BooleanConcept): 2x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): DR-114 junction temperature in °C (NumericalConcept): 1x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; DR-114 maximum ADC INL in LSB (NumericalConcept): 1x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x; DR-114 required flash wait states (NumericalConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (2 runs): Errata applicable to the DR-114 production boards (LabelConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 2x; PO-5521 valid last-time-buy order (BooleanConcept): 2x
- **complex_nda, anthropic/claude-haiku-5-5, default** (3 runs): WORK-1 deletion deadline (DateConcept): 1x
- **complex_nda, azure_ai/ms-decision-model, decision_default** (2 runs): Protected assets at the April snapshot (LabelConcept): 2x; People permitted LUMEN access at the September snapshot (LabelConcept): 2x
- **complex_nda, openai/gpt-6-luna, decision_default** (2 runs): Protected assets at the April snapshot (LabelConcept): 2x; Artifacts permitted retention at the December snapshot (LabelConcept): 2x
- **complex_nda, typesafe/jev-latest, decision_default** (2 runs): Protected assets at the April snapshot (LabelConcept): 2x; Artifacts permitted retention at the December snapshot (LabelConcept): 2x; People permitted LUMEN access at the September snapshot (LabelConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 3x; Water content of V20-2604 after 12 months at the long-term condition in % (NumericalConcept): 1x
- **complex_stability, azure_ai/ms-decision-model, decision_default** (2 runs): Batches with a significant change at the accelerated condition (LabelConcept): 2x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 2x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 2x
- **complex_stability, openai/gpt-6-luna, concurrent** (3 runs): Shelf life supported for the Alu/Alu blister by the rules of section 6 in months (NumericalConcept): 1x; Number of results outside the shelf-life specification (NumericalConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, openai/gpt-6-luna, decision_default** (2 runs): Batches with a significant change at the accelerated condition (LabelConcept): 2x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 2x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 2x
- **complex_stability, typesafe/jev-latest, decision_default** (2 runs): The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 2x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 2x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_dpa, anthropic/claude-haiku-5-5, default**: Amendment effectiveness and precedence: F1 0.89 (P 1.00, R 0.80); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 0.95 (P 0.98, R 0.93); Retention TR-27: F1 0.85 (P 0.91, R 0.81); Supplier approvals in May 2027: F1 0.92 (P 0.95, R 0.89); Liability LC-27: F1 0.85 (P 1.00, R 0.73)
- **complex_dpa, azure_ai/ms-decision-model, decision_default**: Amendment effectiveness and precedence: F1 0.80 (P 0.67, R 1.00); Personal data breach notification: F1 0.76 (P 0.62, R 1.00); Incident IR-27: F1 0.60 (P 0.44, R 0.93); Retention TR-27: F1 0.66 (P 0.49, R 1.00); Supplier approvals in May 2027: F1 0.98 (P 1.00, R 0.95); Liability LC-27: F1 0.83 (P 0.72, R 1.00)
- **complex_dpa, openai/gpt-6-luna, concurrent**: Amendment effectiveness and precedence: F1 0.81 (P 0.86, R 0.80); Personal data breach notification: F1 0.93 (P 0.89, R 1.00); Incident IR-27: F1 1.00 (P 1.00, R 1.00); Retention TR-27: F1 1.00 (P 1.00, R 1.00); Supplier approvals in May 2027: F1 0.98 (P 0.97, R 1.00); Liability LC-27: F1 0.98 (P 1.00, R 0.97)
- **complex_dpa, openai/gpt-6-luna, decision_default**: Amendment effectiveness and precedence: F1 0.80 (P 0.80, R 0.80); Personal data breach notification: F1 0.89 (P 0.80, R 1.00); Incident IR-27: F1 0.74 (P 0.77, R 0.71); Retention TR-27: F1 0.82 (P 0.90, R 0.75); Supplier approvals in May 2027: F1 0.74 (P 0.93, R 0.62); Liability LC-27: F1 0.95 (P 1.00, R 0.90)
- **complex_dpa, openai/gpt-6-luna, default**: Amendment effectiveness and precedence: F1 0.89 (P 1.00, R 0.80); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 1.00 (P 1.00, R 1.00); Retention TR-27: F1 1.00 (P 1.00, R 1.00); Supplier approvals in May 2027: F1 0.98 (P 0.97, R 1.00); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_dpa, openai/gpt-6.1-sol, default**: Amendment effectiveness and precedence: F1 0.77 (P 0.62, R 1.00); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 0.93 (P 0.88, R 1.00); Retention TR-27: F1 0.92 (P 0.86, R 1.00); Supplier approvals in May 2027: F1 0.88 (P 0.78, R 1.00); Liability LC-27: F1 0.87 (P 0.77, R 1.00)
- **complex_dpa, typesafe/jev-latest, decision_default**: Amendment effectiveness and precedence: F1 0.62 (P 0.45, R 1.00); Personal data breach notification: F1 0.84 (P 0.73, R 1.00); Incident IR-27: F1 0.78 (P 0.67, R 0.93); Retention TR-27: F1 0.73 (P 0.80, R 0.67); Supplier approvals in May 2027: F1 0.76 (P 1.00, R 0.62); Liability LC-27: F1 0.93 (P 0.90, R 0.95)
- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.68 (P 0.54, R 0.94); Flash memory specifications: F1 0.93 (P 0.91, R 0.95); DR-114 clock configuration: F1 0.67 (P 0.71, R 0.64); DR-114 thermal assessment: F1 0.73 (P 0.66, R 0.83); DR-114 errata and firmware: F1 0.61 (P 0.66, R 0.58); DR-115 errata and PLL1 limits: F1 0.42 (P 0.78, R 0.30); Revision B end of life: F1 0.79 (P 0.73, R 0.86)
- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.38 (P 0.23, R 1.00); Flash memory specifications: F1 0.58 (P 0.53, R 0.64); DR-114 clock configuration: F1 0.67 (P 0.60, R 0.76); DR-114 thermal assessment: F1 0.48 (P 0.46, R 0.50); DR-114 errata and firmware: F1 0.57 (P 0.64, R 0.52); DR-115 errata and PLL1 limits: F1 0.43 (P 0.59, R 0.34); Revision B end of life: F1 0.83 (P 0.80, R 0.86)
- **complex_mcu, openai/gpt-6-luna, concurrent**: Device limitations of revision C: F1 1.00 (P 1.00, R 1.00); Flash memory specifications: F1 0.93 (P 0.88, R 1.00); DR-114 clock configuration: F1 0.77 (P 0.67, R 0.92); DR-114 thermal assessment: F1 0.92 (P 0.86, R 1.00); DR-114 errata and firmware: F1 0.94 (P 0.89, R 1.00); DR-115 errata and PLL1 limits: F1 0.74 (P 0.59, R 0.98); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.87 (P 0.91, R 0.83); Flash memory specifications: F1 0.64 (P 0.64, R 0.64); DR-114 clock configuration: F1 0.67 (P 0.70, R 0.64); DR-114 thermal assessment: F1 0.37 (P 0.55, R 0.29); DR-114 errata and firmware: F1 0.58 (P 0.69, R 0.50); DR-115 errata and PLL1 limits: F1 0.37 (P 0.57, R 0.28); Revision B end of life: F1 0.92 (P 1.00, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.89 (P 0.80, R 1.00); Flash memory specifications: F1 0.98 (P 0.96, R 1.00); DR-114 clock configuration: F1 0.92 (P 0.94, R 0.90); DR-114 thermal assessment: F1 0.81 (P 0.69, R 0.98); DR-114 errata and firmware: F1 0.98 (P 0.97, R 0.98); DR-115 errata and PLL1 limits: F1 0.89 (P 0.82, R 1.00); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, openai/gpt-6.1-sol, default**: Device limitations of revision C: F1 1.00 (P 1.00, R 1.00); Flash memory specifications: F1 0.97 (P 0.93, R 1.00); DR-114 clock configuration: F1 0.96 (P 0.92, R 1.00); DR-114 thermal assessment: F1 0.95 (P 0.91, R 1.00); DR-114 errata and firmware: F1 0.88 (P 0.79, R 1.00); DR-115 errata and PLL1 limits: F1 0.84 (P 0.72, R 1.00); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.89 (P 0.95, R 0.83); Flash memory specifications: F1 0.72 (P 0.82, R 0.64); DR-114 clock configuration: F1 0.68 (P 0.90, R 0.55); DR-114 thermal assessment: F1 0.45 (P 0.70, R 0.33); DR-114 errata and firmware: F1 0.61 (P 0.91, R 0.45); DR-115 errata and PLL1 limits: F1 0.42 (P 0.84, R 0.28); Revision B end of life: F1 0.77 (P 0.83, R 0.71)
- **complex_nda, anthropic/claude-haiku-5-5, default**: NDA information protection: F1 0.93 (P 0.87, R 1.00); NDA recipient permissions: F1 0.99 (P 0.98, R 1.00); NDA compulsory disclosure Q-27: F1 0.92 (P 0.96, R 0.89); NDA exit obligations X-27: F1 0.96 (P 0.98, R 0.95)
- **complex_nda, azure_ai/ms-decision-model, decision_default**: NDA information protection: F1 0.64 (P 0.47, R 1.00); NDA recipient permissions: F1 0.91 (P 0.83, R 1.00); NDA compulsory disclosure Q-27: F1 0.77 (P 0.62, R 1.00); NDA exit obligations X-27: F1 0.76 (P 0.65, R 0.93)
- **complex_nda, openai/gpt-6-luna, concurrent**: NDA information protection: F1 0.97 (P 0.93, R 1.00); NDA recipient permissions: F1 0.97 (P 0.94, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 0.97 (P 0.94, R 1.00)
- **complex_nda, openai/gpt-6-luna, decision_default**: NDA information protection: F1 0.82 (P 0.88, R 0.78); NDA recipient permissions: F1 0.97 (P 1.00, R 0.93); NDA compulsory disclosure Q-27: F1 0.90 (P 0.82, R 1.00); NDA exit obligations X-27: F1 0.88 (P 1.00, R 0.79)
- **complex_nda, openai/gpt-6-luna, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 0.98 (P 0.96, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, openai/gpt-6.1-sol, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 1.00 (P 1.00, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, typesafe/jev-latest, decision_default**: NDA information protection: F1 0.70 (P 0.64, R 0.78); NDA recipient permissions: F1 0.93 (P 1.00, R 0.87); NDA compulsory disclosure Q-27: F1 0.89 (P 0.89, R 0.89); NDA exit obligations X-27: F1 0.75 (P 1.00, R 0.61)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 0.96 (P 0.93, R 1.00); Stability results of V20-2604: F1 0.95 (P 1.00, R 0.90); Results outside the shelf-life specification: F1 0.70 (P 0.93, R 0.56); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, azure_ai/ms-decision-model, decision_default**: Shelf-life acceptance criteria: F1 0.15 (P 0.11, R 0.21); Stability results of V20-2604: F1 0.39 (P 0.38, R 0.40); Results outside the shelf-life specification: F1 0.41 (P 0.35, R 0.50); Shelf-life rules and proposal: F1 0.54 (P 0.39, R 0.88)
- **complex_stability, openai/gpt-6-luna, concurrent**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.99 (P 0.98, R 1.00); Results outside the shelf-life specification: F1 0.95 (P 0.93, R 0.97); Shelf-life rules and proposal: F1 0.89 (P 0.82, R 1.00)
- **complex_stability, openai/gpt-6-luna, decision_default**: Shelf-life acceptance criteria: F1 0.18 (P 0.25, R 0.14); Stability results of V20-2604: F1 0.33 (P 1.00, R 0.20); Results outside the shelf-life specification: F1 0.56 (P 1.00, R 0.38); Shelf-life rules and proposal: F1 0.58 (P 0.44, R 0.88)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.93 (P 0.91, R 0.95); Shelf-life rules and proposal: F1 0.84 (P 0.73, R 1.00)
- **complex_stability, openai/gpt-6.1-sol, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.98 (P 0.95, R 1.00); Results outside the shelf-life specification: F1 0.96 (P 1.00, R 0.92); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, typesafe/jev-latest, decision_default**: Shelf-life acceptance criteria: F1 0.12 (P 0.11, R 0.14); Stability results of V20-2604: F1 0.26 (P 1.00, R 0.15); Results outside the shelf-life specification: F1 0.47 (P 1.00, R 0.31); Shelf-life rules and proposal: F1 0.67 (P 0.54, R 0.88)

## Errors and warnings

- none
