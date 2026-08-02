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

## ResNet18

Architecture:
- Pretrained ResNet18 (ImageNet)
- Final fully connected layer replaced for binary classification (2 classes)
- Fine-tuned all layers

Hyperparameters:
- Epochs: 5
- Batch Size: 64
- Learning Rate: 1e-3
- Optimizer: Adam
- Input Size: 224 × 224

Results:
- Accuracy: 96.68%
- Precision: 95.63%
- Recall: 97.83%
- F1 Score: 96.72%

Confusion Matrix:
[[9553, 447],
 [217, 9783]]

Notes:
- Achieved the best performance among all CNN-based models tested.
- Improved accuracy by 3.55% over the reproduced Paper CNN baseline.
- Improved F1 score by 2.75% over the Paper CNN and 2.75% over the Improved CNN.
- High recall indicates the model successfully identified most images belonging to the positive class.
- Transfer learning with ImageNet-pretrained weights significantly improved performance compared to training CNNs from scratch.

## Vision Transformer (ViT-B/16)

Architecture:
- Pretrained ViT-B/16 (ImageNet)
- Classification head replaced for binary classification

Hyperparameters:
- Epochs: 1
- Batch Size: 64
- Learning Rate: 1e-4
- Optimizer: Adam

Results:
- Accuracy: 97.15%
- Precision: 96.87%
- Recall: 97.45%
- F1 Score: 97.16%

Confusion Matrix:
[[9685, 315],
 [255, 9745]]

Notes:
- Highest performing model among all evaluated architectures.
- Fine-tuned pretrained ImageNet weights.
- Achieved state-of-the-art performance within this benchmark despite training for only one epoch.

