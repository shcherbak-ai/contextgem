# Decision threshold sweep: stability_2026-10-20_text

## Aspect F1 by threshold

Mean of the cases. The second line of a model has the precision and the recall. Only the models recorded on each case are shown.

| Model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0.287 | 0.337 | 0.370 | 0.367 | 0.378 | 0.334 | 0.382 | 0.447 | 0.483 | 0.483 at 0.90 (within 0.01: 0.90-0.90) |
| | 0.17 / 1.00 | 0.21 / 0.96 | 0.25 / 0.76 | 0.27 / 0.62 | 0.30 / 0.52 | 0.35 / 0.39 | 0.44 / 0.39 | 0.68 / 0.39 | 0.76 / 0.39 | |
| openai/gpt-6-luna | 0.326 | 0.392 | 0.405 | 0.406 | 0.414 | 0.431 | 0.431 | 0.467 | 0.451 | 0.479 at 0.95 (within 0.01: 0.85-0.95) |
| | 0.34 / 0.41 | 0.55 / 0.41 | 0.57 / 0.41 | 0.61 / 0.40 | 0.67 / 0.40 | 0.71 / 0.40 | 0.71 / 0.40 | 0.76 / 0.40 | 0.94 / 0.37 | |
| typesafe/jev-latest | 0.250 | 0.287 | 0.296 | 0.326 | 0.375 | 0.419 | 0.434 | 0.428 | 0.367 | 0.452 at 0.75 (within 0.01: 0.75-0.75) |
| | 0.15 / 0.86 | 0.21 / 0.58 | 0.34 / 0.41 | 0.56 / 0.37 | 0.66 / 0.37 | 0.72 / 0.37 | 0.76 / 0.37 | 0.88 / 0.34 | 1.00 / 0.27 | |
| **Mean of the models** | 0.288 | 0.339 | 0.357 | 0.367 | 0.389 | 0.395 | 0.416 | 0.447 | 0.434 | 0.452 at 0.85 |

## Best threshold of each case

With the F1 at it, and the F1 at 0.5, in brackets.

- **azure_ai/ms-decision-model**: complex_stability: 0.90 (0.483, 0.378)
- **openai/gpt-6-luna**: complex_stability: 0.95 (0.479, 0.414)
- **typesafe/jev-latest**: complex_stability: 0.75 (0.452, 0.375)

## How the probabilities relate to the reference answers

The share of the (paragraph, aspect) pairs in each range of probability that are in the reference answers, with the number of the pairs in brackets. All the cases recorded for a model together, without the pairs that are not scored. For the refused answers, the number of the pairs that are in the reference answers.

| Model | 0.0-0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | 0.9-1.0 | Refused |
|---|---|---|---|---|---|---|---|
| azure_ai/ms-decision-model | 0% (187) | 10% (136) | 17% (64) | 19% (36) | 0% (18) | 81% (21) | 0 of 0 |
| openai/gpt-6-luna | 7% (401) | 0% (21) | 14% (7) | 0% (3) | 22% (9) | 88% (17) | 0 of 4 |
| typesafe/jev-latest | 4% (177) | 11% (223) | 6% (33) | 0% (9) | 50% (10) | 100% (10) | 0 of 0 |
