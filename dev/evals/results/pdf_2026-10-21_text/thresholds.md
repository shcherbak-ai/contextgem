# Decision threshold sweep: pdf_2026-10-21_text

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.340 | 0.397 | 0.442 | 0.448 | 0.481 | 0.462 | 0.479 | 0.525 | 0.546 | 0.546 at 0.90 (within 0.01: 0.90-0.90) |
| | 0.22 / 0.91 | 0.28 / 0.87 | 0.34 / 0.73 | 0.37 / 0.65 | 0.44 / 0.61 | 0.50 / 0.51 | 0.57 / 0.49 | 0.72 / 0.47 | 0.81 / 0.46 | |
| openai/gpt-6-luna | 0.411 | 0.483 | 0.507 | 0.508 | 0.523 | 0.538 | 0.542 | 0.535 | 0.521 | 0.544 at 0.65 (within 0.01: 0.60-0.85) |
| | 0.37 / 0.54 | 0.54 / 0.53 | 0.59 / 0.52 | 0.63 / 0.49 | 0.70 / 0.49 | 0.75 / 0.48 | 0.78 / 0.48 | 0.84 / 0.44 | 0.96 / 0.41 | |
| typesafe/jev-latest | 0.365 | 0.444 | 0.469 | 0.481 | 0.503 | 0.509 | 0.504 | 0.444 | 0.319 | 0.513 at 0.65 (within 0.01: 0.55-0.70) |
| | 0.25 / 0.89 | 0.36 / 0.67 | 0.48 / 0.54 | 0.63 / 0.48 | 0.73 / 0.46 | 0.81 / 0.42 | 0.86 / 0.40 | 1.00 / 0.33 | 0.93 / 0.23 | |
| **Mean of the models** | 0.372 | 0.441 | 0.473 | 0.479 | 0.502 | 0.503 | 0.508 | 0.501 | 0.462 | 0.509 at 0.75 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_mcu: 0.80 (0.603, 0.562); complex_stability: 0.90 (0.497, 0.399)
- **openai/gpt-6-luna**: complex_mcu: 0.65 (0.658, 0.633); complex_stability: 0.95 (0.479, 0.414)
- **typesafe/jev-latest**: complex_mcu: 0.30 (0.646, 0.630); complex_stability: 0.75 (0.452, 0.375)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 2% (1420) | 9% (370) | 13% (141) | 21% (95) | 18% (62) | 83% (93) | 0 of 0 |
| openai/gpt-6-luna | 4% (1902) | 6% (102) | 19% (36) | 15% (26) | 45% (31) | 95% (74) | 0 of 10 |
| typesafe/jev-latest | 2% (1465) | 12% (512) | 25% (93) | 41% (37) | 85% (46) | 100% (28) | 0 of 0 |
