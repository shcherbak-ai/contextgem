# Decision threshold sweep: baseline_2026-10-15

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.481 | 0.555 | 0.625 | 0.660 | 0.706 | 0.731 | 0.751 | 0.778 | 0.785 | 0.786 at 0.85 (within 0.01: 0.80-0.90) |
| | 0.33 / 0.94 | 0.41 / 0.93 | 0.50 / 0.90 | 0.55 / 0.89 | 0.62 / 0.87 | 0.68 / 0.85 | 0.74 / 0.83 | 0.81 / 0.82 | 0.88 / 0.75 | |
| openai/gpt-6-luna | 0.651 | 0.722 | 0.769 | 0.767 | 0.783 | 0.784 | 0.775 | 0.751 | 0.704 | 0.787 at 0.45 (within 0.01: 0.45-0.65) |
| | 0.54 / 0.86 | 0.66 / 0.83 | 0.74 / 0.81 | 0.77 / 0.78 | 0.84 / 0.75 | 0.88 / 0.73 | 0.90 / 0.70 | 0.94 / 0.66 | 0.96 / 0.59 | |
| typesafe/jev-latest | 0.500 | 0.667 | 0.723 | 0.753 | 0.753 | 0.744 | 0.680 | 0.592 | 0.355 | 0.761 at 0.45 (within 0.01: 0.40-0.50) |
| | 0.36 / 0.96 | 0.56 / 0.90 | 0.66 / 0.86 | 0.75 / 0.81 | 0.81 / 0.76 | 0.88 / 0.69 | 0.91 / 0.59 | 0.96 / 0.47 | 0.84 / 0.25 | |
| **Mean of the models** | 0.544 | 0.648 | 0.706 | 0.727 | 0.747 | 0.753 | 0.735 | 0.707 | 0.615 | 0.753 at 0.60 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_dpa: 0.90 (0.873, 0.771); complex_mcu: 0.80 (0.594, 0.564); complex_nda: 0.90 (0.904, 0.784)
- **openai/gpt-6-luna**: complex_dpa: 0.65 (0.844, 0.823); complex_mcu: 0.65 (0.658, 0.633); complex_nda: 0.30 (0.908, 0.892)
- **typesafe/jev-latest**: complex_dpa: 0.60 (0.801, 0.793); complex_mcu: 0.30 (0.644, 0.633); complex_nda: 0.40 (0.866, 0.834)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 1% (2000) | 5% (348) | 10% (131) | 20% (87) | 26% (92) | 86% (191) | 0 of 0 |
| openai/gpt-6-luna | 3% (2397) | 10% (153) | 27% (55) | 33% (36) | 68% (40) | 96% (139) | 0 of 29 |
| typesafe/jev-latest | 1% (2023) | 8% (494) | 29% (115) | 58% (76) | 87% (91) | 98% (50) | 0 of 0 |
