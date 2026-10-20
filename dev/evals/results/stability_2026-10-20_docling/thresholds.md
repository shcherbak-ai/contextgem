# Decision threshold sweep: stability_2026-10-20_docling

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.392 | 0.471 | 0.554 | 0.610 | 0.642 | 0.670 | 0.698 | 0.746 | 0.826 | 0.849 at 0.95 (within 0.01: 0.95-0.95) |
| | 0.26 / 1.00 | 0.33 / 1.00 | 0.42 / 0.98 | 0.48 / 0.98 | 0.52 / 0.96 | 0.55 / 0.90 | 0.61 / 0.85 | 0.72 / 0.83 | 0.88 / 0.83 | |
| openai/gpt-6-luna | 0.594 | 0.666 | 0.683 | 0.716 | 0.742 | 0.764 | 0.781 | 0.793 | 0.804 | 0.847 at 0.95 (within 0.01: 0.95-0.95) |
| | 0.48 / 0.94 | 0.55 / 0.94 | 0.58 / 0.92 | 0.64 / 0.90 | 0.66 / 0.90 | 0.70 / 0.90 | 0.74 / 0.88 | 0.79 / 0.87 | 0.84 / 0.81 | |
| typesafe/jev-latest | 0.337 | 0.453 | 0.555 | 0.666 | 0.746 | 0.792 | 0.825 | 0.773 | 0.636 | 0.825 at 0.70 (within 0.01: 0.70-0.75) |
| | 0.23 / 1.00 | 0.36 / 0.98 | 0.45 / 0.94 | 0.56 / 0.91 | 0.71 / 0.86 | 0.86 / 0.82 | 0.94 / 0.79 | 0.96 / 0.69 | 1.00 / 0.53 | |
| **Mean of the models** | 0.441 | 0.530 | 0.597 | 0.664 | 0.710 | 0.742 | 0.768 | 0.771 | 0.756 | 0.782 at 0.85 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_stability: 0.95 (0.849, 0.642)
- **openai/gpt-6-luna**: complex_stability: 0.95 (0.847, 0.742)
- **typesafe/jev-latest**: complex_stability: 0.70 (0.825, 0.746)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0% (263) | 1% (72) | 4% (28) | 14% (35) | 5% (19) | 89% (45) | 0 of 0 |
| openai/gpt-6-luna | 1% (354) | 4% (27) | 6% (16) | 11% (9) | 33% (12) | 86% (44) | 0 of 0 |
| typesafe/jev-latest | 0% (176) | 2% (166) | 8% (65) | 19% (16) | 82% (11) | 100% (28) | 0 of 0 |
