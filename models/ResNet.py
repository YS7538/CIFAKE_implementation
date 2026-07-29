import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Model(nn.Module):

    def __init__(self):
        super().__init__()

        # Load ImageNet pretrained weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace final classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)