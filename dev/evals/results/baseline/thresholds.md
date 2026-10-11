# Decision threshold sweep: baseline

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.527 | 0.610 | 0.673 | 0.727 | 0.769 | 0.820 | 0.836 | 0.869 | 0.877 | 0.879 at 0.85 (within 0.01: 0.85-0.90) |
| | 0.37 / 1.00 | 0.45 / 1.00 | 0.52 / 0.99 | 0.59 / 0.98 | 0.65 / 0.98 | 0.73 / 0.97 | 0.76 / 0.95 | 0.83 / 0.94 | 0.91 / 0.87 | |
| openai/gpt-6-luna | 0.709 | 0.792 | 0.835 | 0.844 | 0.849 | 0.840 | 0.843 | 0.803 | 0.747 | 0.854 at 0.35 (within 0.01: 0.35-0.65) |
| | 0.58 / 0.98 | 0.69 / 0.95 | 0.77 / 0.93 | 0.81 / 0.89 | 0.87 / 0.84 | 0.89 / 0.81 | 0.92 / 0.79 | 0.94 / 0.72 | 0.97 / 0.63 | |
| strands_decider/kev-4b | 0.392 | 0.536 | 0.655 | 0.714 | 0.747 | 0.724 | 0.701 | 0.486 | 0.199 | 0.750 at 0.55 (within 0.01: 0.50-0.55) |
| | 0.25 / 0.99 | 0.39 / 0.97 | 0.54 / 0.92 | 0.63 / 0.88 | 0.72 / 0.82 | 0.79 / 0.71 | 0.90 / 0.61 | 0.93 / 0.35 | 0.62 / 0.15 | |
| strands_decider/nimble | 0.489 | 0.626 | 0.635 | 0.664 | 0.695 | 0.751 | 0.774 | 0.725 | 0.711 | 0.774 at 0.70 (within 0.01: 0.70-0.70) |
| | 0.37 / 0.84 | 0.57 / 0.82 | 0.62 / 0.77 | 0.66 / 0.76 | 0.70 / 0.74 | 0.79 / 0.74 | 0.87 / 0.72 | 0.87 / 0.66 | 0.98 / 0.59 | |
| typesafe/jev-latest | 0.515 | 0.703 | 0.765 | 0.807 | 0.816 | 0.812 | 0.748 | 0.662 | 0.403 | 0.818 at 0.55 (within 0.01: 0.45-0.60) |
| | 0.36 / 1.00 | 0.59 / 0.98 | 0.68 / 0.94 | 0.76 / 0.91 | 0.82 / 0.85 | 0.85 / 0.81 | 0.89 / 0.68 | 0.94 / 0.54 | 0.83 / 0.29 | |
| **Mean of the models** | 0.526 | 0.653 | 0.713 | 0.751 | 0.775 | 0.789 | 0.780 | 0.709 | 0.587 | 0.789 at 0.60 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_dpa: 0.85 (0.866, 0.762); complex_nda: 0.90 (0.902, 0.775)
- **openai/gpt-6-luna**: complex_dpa: 0.75 (0.837, 0.832); complex_nda: 0.35 (0.889, 0.866)
- **strands_decider/kev-4b**: complex_dpa: 0.55 (0.730, 0.725); complex_nda: 0.55 (0.770, 0.768)
- **strands_decider/nimble**: complex_dpa: 0.85 (0.819, 0.697); complex_nda: 0.70 (0.753, 0.692)
- **typesafe/jev-latest**: complex_dpa: 0.55 (0.794, 0.787); complex_nda: 0.40 (0.857, 0.846)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0% (773) | 1% (109) | 2% (49) | 17% (36) | 26% (43) | 88% (120) | 0 of 0 |
| openai/gpt-6-luna | 1% (884) | 10% (70) | 36% (33) | 50% (16) | 76% (25) | 96% (78) | 0 of 24 |
| strands_decider/kev-4b | 0% (598) | 4% (312) | 19% (84) | 47% (57) | 88% (64) | 93% (15) | 0 of 0 |
| strands_decider/nimble | 3% (784) | 7% (153) | 7% (58) | 8% (37) | 61% (28) | 97% (70) | 0 of 0 |
| typesafe/jev-latest | 0% (740) | 4% (201) | 23% (56) | 58% (43) | 82% (57) | 97% (33) | 0 of 0 |
