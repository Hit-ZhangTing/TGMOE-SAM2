import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True, out_indices=[1,2,3,4]):
        super().__init__()
        self.out_indices = out_indices

        # Load pretrained ResNet-50
        backbone = resnet50(pretrained=pretrained)

        # Keep layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        # ResNet stages
        self.layer1 = backbone.layer1  # C2  256
        self.layer2 = backbone.layer2  # C3  512
        self.layer3 = backbone.layer3  # C4 1024
        self.layer4 = backbone.layer4  # C5 2048

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        # channel list (must match extracted layers)
        self.channel_list = [256, 512, 1024, 2048]

    def forward(self, x):
        x = self.stem(x)
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx+1 in self.out_indices:
                outs.append(x)

        return outs
