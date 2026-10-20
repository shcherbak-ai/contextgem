# Decision threshold sweep: docling_2026-10-20_docling

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.560 | 0.644 | 0.701 | 0.749 | 0.770 | 0.801 | 0.826 | 0.848 | 0.828 | 0.848 at 0.80 (within 0.01: 0.80-0.80) |
| | 0.41 / 0.96 | 0.51 / 0.95 | 0.58 / 0.93 | 0.66 / 0.92 | 0.70 / 0.91 | 0.75 / 0.91 | 0.80 / 0.90 | 0.86 / 0.88 | 0.89 / 0.82 | |
| openai/gpt-6-luna | 0.769 | 0.821 | 0.849 | 0.862 | 0.863 | 0.865 | 0.863 | 0.866 | 0.837 | 0.873 at 0.55 (within 0.01: 0.50-0.80) |
| | 0.67 / 0.92 | 0.76 / 0.92 | 0.82 / 0.90 | 0.85 / 0.90 | 0.86 / 0.89 | 0.88 / 0.86 | 0.89 / 0.85 | 0.91 / 0.84 | 0.92 / 0.79 | |
| typesafe/jev-latest | 0.622 | 0.747 | 0.783 | 0.820 | 0.808 | 0.786 | 0.759 | 0.667 | 0.499 | 0.820 at 0.40 (within 0.01: 0.40-0.55) |
| | 0.48 / 0.98 | 0.66 / 0.94 | 0.74 / 0.90 | 0.82 / 0.87 | 0.86 / 0.81 | 0.90 / 0.75 | 0.94 / 0.70 | 0.98 / 0.56 | 0.89 / 0.39 | |
| **Mean of the models** | 0.650 | 0.737 | 0.778 | 0.810 | 0.814 | 0.817 | 0.816 | 0.794 | 0.721 | 0.824 at 0.55 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_dpa: 0.80 (0.897, 0.771); complex_mcu: 0.80 (0.749, 0.708); complex_nda: 0.95 (0.934, 0.830)
- **openai/gpt-6-luna**: complex_dpa: 0.55 (0.915, 0.908); complex_mcu: 0.70 (0.755, 0.749); complex_nda: 0.55 (0.958, 0.932)
- **typesafe/jev-latest**: complex_dpa: 0.55 (0.870, 0.844); complex_mcu: 0.25 (0.761, 0.689); complex_nda: 0.55 (0.905, 0.889)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 1% (2178) | 5% (252) | 10% (84) | 8% (51) | 45% (66) | 86% (218) | 0 of 0 |
| openai/gpt-6-luna | 1% (2476) | 15% (75) | 17% (24) | 41% (27) | 55% (33) | 91% (207) | 0 of 7 |
| typesafe/jev-latest | 0% (2244) | 12% (284) | 36% (98) | 57% (65) | 90% (83) | 99% (75) | 0 of 0 |
