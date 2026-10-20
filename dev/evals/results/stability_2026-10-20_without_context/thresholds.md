# Decision threshold sweep: stability_2026-10-20_without_context

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.257 | 0.310 | 0.373 | 0.413 | 0.438 | 0.454 | 0.504 | 0.518 | 0.465 | 0.537 at 0.85 (within 0.01: 0.75-0.85) |
| | 0.15 / 1.00 | 0.19 / 1.00 | 0.23 / 0.95 | 0.27 / 0.93 | 0.29 / 0.88 | 0.33 / 0.76 | 0.42 / 0.72 | 0.58 / 0.59 | 0.68 / 0.41 | |
| openai/gpt-6-luna | 0.380 | 0.448 | 0.424 | 0.439 | 0.437 | 0.424 | 0.424 | 0.467 | 0.451 | 0.479 at 0.95 (within 0.01: 0.85-0.95) |
| | 0.32 / 0.59 | 0.42 / 0.59 | 0.41 / 0.54 | 0.45 / 0.51 | 0.60 / 0.46 | 0.67 / 0.40 | 0.67 / 0.40 | 0.76 / 0.40 | 0.94 / 0.37 | |
| typesafe/jev-latest | 0.244 | 0.318 | 0.421 | 0.419 | 0.454 | 0.434 | 0.447 | 0.428 | 0.367 | 0.472 at 0.75 (within 0.01: 0.75-0.75) |
| | 0.15 / 0.95 | 0.20 / 0.95 | 0.29 / 0.84 | 0.31 / 0.63 | 0.53 / 0.46 | 0.76 / 0.37 | 0.78 / 0.37 | 0.88 / 0.34 | 1.00 / 0.27 | |
| **Mean of the models** | 0.294 | 0.359 | 0.406 | 0.423 | 0.443 | 0.437 | 0.458 | 0.471 | 0.428 | 0.481 at 0.75 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_stability: 0.85 (0.537, 0.438)
- **openai/gpt-6-luna**: complex_stability: 0.95 (0.479, 0.437)
- **typesafe/jev-latest**: complex_stability: 0.75 (0.472, 0.454)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0% (144) | 2% (126) | 6% (53) | 15% (53) | 28% (60) | 69% (26) | 0 of 0 |
| openai/gpt-6-luna | 6% (375) | 8% (25) | 18% (22) | 22% (9) | 20% (10) | 88% (17) | 0 of 4 |
| typesafe/jev-latest | 2% (124) | 2% (199) | 20% (98) | 32% (22) | 56% (9) | 100% (10) | 0 of 0 |
