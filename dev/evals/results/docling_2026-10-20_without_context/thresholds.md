# Decision threshold sweep: docling_2026-10-20_without_context

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.425 | 0.491 | 0.550 | 0.577 | 0.613 | 0.634 | 0.655 | 0.665 | 0.633 | 0.665 at 0.80 (within 0.01: 0.70-0.85) |
| | 0.30 / 0.88 | 0.38 / 0.87 | 0.46 / 0.83 | 0.50 / 0.81 | 0.57 / 0.78 | 0.62 / 0.75 | 0.68 / 0.75 | 0.73 / 0.71 | 0.75 / 0.63 | |
| openai/gpt-6-luna | 0.540 | 0.598 | 0.637 | 0.646 | 0.659 | 0.666 | 0.664 | 0.615 | 0.611 | 0.672 at 0.65 (within 0.01: 0.45-0.70) |
| | 0.43 / 0.76 | 0.53 / 0.70 | 0.61 / 0.68 | 0.66 / 0.65 | 0.75 / 0.61 | 0.79 / 0.59 | 0.85 / 0.57 | 0.91 / 0.50 | 0.98 / 0.49 | |
| typesafe/jev-latest | 0.505 | 0.667 | 0.705 | 0.719 | 0.692 | 0.681 | 0.635 | 0.526 | 0.331 | 0.720 at 0.45 (within 0.01: 0.35-0.45) |
| | 0.36 / 0.97 | 0.56 / 0.89 | 0.65 / 0.82 | 0.72 / 0.74 | 0.78 / 0.66 | 0.86 / 0.60 | 0.90 / 0.54 | 0.93 / 0.41 | 0.86 / 0.24 | |
| **Mean of the models** | 0.490 | 0.585 | 0.631 | 0.647 | 0.655 | 0.660 | 0.652 | 0.602 | 0.525 | 0.665 at 0.45 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_mcu: 0.80 (0.665, 0.613)
- **openai/gpt-6-luna**: complex_mcu: 0.65 (0.672, 0.659)
- **typesafe/jev-latest**: complex_mcu: 0.45 (0.720, 0.692)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 2% (1225) | 4% (201) | 10% (79) | 10% (48) | 34% (56) | 70% (110) | 0 of 0 |
| openai/gpt-6-luna | 3% (1467) | 15% (86) | 21% (39) | 26% (27) | 48% (21) | 97% (63) | 0 of 16 |
| typesafe/jev-latest | 0% (1271) | 8% (262) | 35% (72) | 49% (35) | 81% (57) | 100% (22) | 0 of 0 |
