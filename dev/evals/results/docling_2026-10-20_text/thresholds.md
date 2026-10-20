# Decision threshold sweep: docling_2026-10-20_text

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.485 | 0.558 | 0.623 | 0.653 | 0.700 | 0.736 | 0.747 | 0.782 | 0.794 | 0.794 at 0.90 (within 0.01: 0.90-0.90) |
| | 0.34 / 0.94 | 0.41 / 0.93 | 0.50 / 0.90 | 0.54 / 0.88 | 0.61 / 0.87 | 0.69 / 0.85 | 0.73 / 0.83 | 0.81 / 0.81 | 0.89 / 0.77 | |
| openai/gpt-6-luna | 0.651 | 0.722 | 0.769 | 0.767 | 0.783 | 0.784 | 0.775 | 0.751 | 0.704 | 0.787 at 0.45 (within 0.01: 0.45-0.65) |
| | 0.54 / 0.86 | 0.66 / 0.83 | 0.74 / 0.81 | 0.77 / 0.78 | 0.84 / 0.75 | 0.88 / 0.73 | 0.90 / 0.70 | 0.94 / 0.66 | 0.96 / 0.59 | |
| typesafe/jev-latest | 0.497 | 0.667 | 0.725 | 0.757 | 0.755 | 0.736 | 0.693 | 0.598 | 0.362 | 0.757 at 0.40 (within 0.01: 0.40-0.55) |
| | 0.35 / 0.96 | 0.56 / 0.89 | 0.66 / 0.86 | 0.75 / 0.81 | 0.81 / 0.76 | 0.86 / 0.69 | 0.92 / 0.60 | 0.96 / 0.47 | 0.84 / 0.26 | |
| **Mean of the models** | 0.544 | 0.649 | 0.706 | 0.726 | 0.746 | 0.752 | 0.738 | 0.710 | 0.620 | 0.752 at 0.60 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_dpa: 0.90 (0.886, 0.763); complex_mcu: 0.80 (0.603, 0.556); complex_nda: 0.90 (0.908, 0.781)
- **openai/gpt-6-luna**: complex_dpa: 0.65 (0.844, 0.823); complex_mcu: 0.65 (0.658, 0.633); complex_nda: 0.30 (0.908, 0.892)
- **typesafe/jev-latest**: complex_dpa: 0.60 (0.797, 0.787); complex_mcu: 0.35 (0.653, 0.628); complex_nda: 0.40 (0.866, 0.851)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 1% (2008) | 6% (339) | 10% (125) | 18% (95) | 24% (88) | 87% (194) | 0 of 0 |
| openai/gpt-6-luna | 3% (2397) | 10% (153) | 27% (55) | 33% (36) | 68% (40) | 96% (139) | 0 of 29 |
| typesafe/jev-latest | 1% (2031) | 8% (485) | 30% (114) | 55% (76) | 88% (92) | 98% (51) | 0 of 0 |
