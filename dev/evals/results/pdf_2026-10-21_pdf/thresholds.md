# Decision threshold sweep: pdf_2026-10-21_pdf

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.437 | 0.506 | 0.567 | 0.597 | 0.632 | 0.676 | 0.715 | 0.733 | 0.751 | 0.751 at 0.90 (within 0.01: 0.85-0.95) |
| | 0.31 / 0.95 | 0.39 / 0.92 | 0.46 / 0.90 | 0.51 / 0.88 | 0.56 / 0.87 | 0.62 / 0.85 | 0.69 / 0.83 | 0.77 / 0.78 | 0.86 / 0.74 | |
| openai/gpt-6-luna | 0.638 | 0.667 | 0.685 | 0.706 | 0.728 | 0.737 | 0.745 | 0.747 | 0.746 | 0.756 at 0.95 (within 0.01: 0.75-0.95) |
| | 0.54 / 0.89 | 0.58 / 0.88 | 0.61 / 0.86 | 0.65 / 0.84 | 0.68 / 0.84 | 0.69 / 0.84 | 0.71 / 0.82 | 0.75 / 0.80 | 0.81 / 0.75 | |
| typesafe/jev-latest | 0.482 | 0.586 | 0.637 | 0.705 | 0.741 | 0.738 | 0.724 | 0.659 | 0.462 | 0.743 at 0.55 (within 0.01: 0.50-0.65) |
| | 0.36 / 0.98 | 0.51 / 0.92 | 0.58 / 0.87 | 0.69 / 0.84 | 0.80 / 0.78 | 0.86 / 0.72 | 0.92 / 0.66 | 0.93 / 0.57 | 0.86 / 0.38 | |
| **Mean of the models** | 0.519 | 0.586 | 0.630 | 0.669 | 0.700 | 0.717 | 0.728 | 0.713 | 0.653 | 0.728 at 0.70 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_mcu: 0.80 (0.721, 0.641); complex_stability: 0.95 (0.835, 0.622)
- **openai/gpt-6-luna**: complex_mcu: 0.75 (0.745, 0.730); complex_stability: 0.95 (0.837, 0.726)
- **typesafe/jev-latest**: complex_mcu: 0.35 (0.771, 0.722); complex_stability: 0.75 (0.833, 0.760)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 1% (1585) | 6% (241) | 11% (80) | 7% (73) | 31% (52) | 81% (150) | 0 of 0 |
| openai/gpt-6-luna | 2% (1864) | 11% (62) | 14% (35) | 17% (24) | 39% (49) | 82% (147) | 0 of 0 |
| typesafe/jev-latest | 1% (1591) | 10% (301) | 15% (123) | 54% (50) | 79% (67) | 100% (49) | 0 of 0 |
