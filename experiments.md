# CIFAKE Experiments

## Paper CNN

Date: 2026-07-25

Architecture:
- Conv(32)
- Conv(32)
- FC(64)

Hyperparameters:
- Epochs: 5
- Batch Size: 64
- Learning Rate: 1e-3
- Optimizer: Adam

Results:
- Accuracy: 93.13%
- Precision: 94.36%
- Recall: 91.73%
- F1 Score: 93.03%

Confusion Matrix:
[[9452, 548],
 [827, 9173]]

Notes:
- Successfully reproduced the paper baseline.
- Roughly 1% lower F1 than Improved CNN.
- Smaller model with fewer parameters.

## Improved CNN

Results:
- Accuracy: 94.05%
- Precision: 95.32%
- Recall: 92.65%
- F1: 93.97%

Notes:
- Increased second convolution filters (64).
- Increased FC layer to 128 neurons.
- Consistently outperformed Paper CNN.