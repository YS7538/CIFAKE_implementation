import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)
