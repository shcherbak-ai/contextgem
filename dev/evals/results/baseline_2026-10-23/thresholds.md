# Decision threshold sweep: baseline_2026-10-23

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.434 | 0.500 | 0.559 | 0.581 | 0.626 | 0.646 | 0.660 | 0.700 | 0.713 | 0.713 at 0.90 (within 0.01: 0.90-0.90) |
| | 0.29 / 0.95 | 0.36 / 0.93 | 0.43 / 0.86 | 0.47 / 0.82 | 0.54 / 0.80 | 0.61 / 0.75 | 0.67 / 0.72 | 0.78 / 0.71 | 0.85 / 0.67 | |
| openai/gpt-6-luna | 0.570 | 0.640 | 0.678 | 0.677 | 0.690 | 0.696 | 0.689 | 0.680 | 0.641 | 0.696 at 0.60 (within 0.01: 0.45-0.70) |
| | 0.49 / 0.75 | 0.63 / 0.73 | 0.70 / 0.71 | 0.73 / 0.68 | 0.80 / 0.66 | 0.84 / 0.64 | 0.85 / 0.62 | 0.89 / 0.59 | 0.96 / 0.53 | |
| typesafe/jev-latest | 0.438 | 0.578 | 0.625 | 0.644 | 0.661 | 0.657 | 0.636 | 0.561 | 0.343 | 0.668 at 0.55 (within 0.01: 0.50-0.55) |
| | 0.31 / 0.96 | 0.48 / 0.82 | 0.64 / 0.72 | 0.72 / 0.68 | 0.80 / 0.64 | 0.84 / 0.60 | 0.90 / 0.55 | 0.94 / 0.45 | 0.91 / 0.24 | |
| **Mean of the models** | 0.481 | 0.573 | 0.621 | 0.634 | 0.659 | 0.666 | 0.661 | 0.647 | 0.566 | 0.666 at 0.60 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_dpa: 0.90 (0.881, 0.760); complex_mcu: 0.80 (0.600, 0.559); complex_nda: 0.90 (0.912, 0.781); complex_stability: 0.95 (0.490, 0.406)
- **openai/gpt-6-luna**: complex_dpa: 0.65 (0.844, 0.823); complex_mcu: 0.65 (0.658, 0.633); complex_nda: 0.30 (0.908, 0.892); complex_stability: 0.95 (0.479, 0.414)
- **typesafe/jev-latest**: complex_dpa: 0.55 (0.826, 0.777); complex_mcu: 0.55 (0.654, 0.653); complex_nda: 0.45 (0.858, 0.824); complex_stability: 0.70 (0.446, 0.389)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 1% (2194) | 7% (476) | 11% (189) | 19% (129) | 21% (108) | 86% (215) | 0 of 0 |
| openai/gpt-6-luna | 3% (2798) | 9% (174) | 26% (62) | 31% (39) | 59% (49) | 95% (156) | 0 of 33 |
| typesafe/jev-latest | 1% (2124) | 11% (842) | 22% (112) | 49% (71) | 87% (104) | 98% (58) | 0 of 0 |
