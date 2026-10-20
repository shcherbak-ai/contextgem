# Decision threshold sweep: pdf_2026-10-21_docling

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.445 | 0.520 | 0.586 | 0.641 | 0.674 | 0.703 | 0.725 | 0.752 | 0.761 | 0.761 at 0.85 (within 0.01: 0.80-0.90) |
| | 0.31 / 0.94 | 0.39 / 0.92 | 0.48 / 0.89 | 0.55 / 0.87 | 0.61 / 0.86 | 0.66 / 0.83 | 0.70 / 0.83 | 0.81 / 0.77 | 0.89 / 0.73 | |
| openai/gpt-6-luna | 0.644 | 0.695 | 0.704 | 0.731 | 0.746 | 0.758 | 0.768 | 0.770 | 0.768 | 0.782 at 0.85 (within 0.01: 0.85-0.85) |
| | 0.54 / 0.89 | 0.60 / 0.88 | 0.64 / 0.85 | 0.69 / 0.84 | 0.71 / 0.83 | 0.74 / 0.82 | 0.77 / 0.81 | 0.80 / 0.79 | 0.85 / 0.75 | |
| typesafe/jev-latest | 0.498 | 0.604 | 0.654 | 0.717 | 0.730 | 0.691 | 0.697 | 0.644 | 0.476 | 0.730 at 0.50 (within 0.01: 0.45-0.55) |
| | 0.38 / 0.97 | 0.53 / 0.92 | 0.61 / 0.86 | 0.71 / 0.83 | 0.80 / 0.76 | 0.87 / 0.67 | 0.96 / 0.62 | 0.98 / 0.56 | 0.86 / 0.39 | |
| **Mean of the models** | 0.529 | 0.606 | 0.648 | 0.697 | 0.716 | 0.717 | 0.730 | 0.722 | 0.668 | 0.730 at 0.70 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_mcu: 0.75 (0.743, 0.697); complex_stability: 0.90 (0.854, 0.650)
- **openai/gpt-6-luna**: complex_mcu: 0.70 (0.755, 0.749); complex_stability: 0.95 (0.847, 0.742)
- **typesafe/jev-latest**: complex_mcu: 0.35 (0.769, 0.693); complex_stability: 0.65 (0.830, 0.767)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 1% (1633) | 5% (219) | 10% (87) | 9% (46) | 39% (62) | 85% (134) | 0 of 0 |
| openai/gpt-6-luna | 2% (1874) | 14% (63) | 11% (35) | 21% (24) | 40% (35) | 86% (145) | 0 of 5 |
| typesafe/jev-latest | 1% (1620) | 10% (301) | 26% (113) | 54% (56) | 92% (40) | 100% (51) | 0 of 0 |
