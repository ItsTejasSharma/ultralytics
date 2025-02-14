# ultralytics/nn/modules/bifpn.py
import torch
import torch.nn as nn  # Import the torch.nn module as nn
import torch.nn.functional as F
from .block import BiFPNBlock, DepthwiseConvBlock, ConvBlock  # Import from block.py


class BiFPN(nn.Module):
    def __init__(self, size, feature_size=64, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.p3 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)

        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(size[2], feature_size, kernel_size=3, stride=2, padding=1)

        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = ConvBlock(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.bifpns = nn.ModuleList([BiFPNBlock(feature_size) for _ in range(num_layers)]) # Use ModuleList


    def forward(self, inputs):
        c3, c4, c5 = inputs

        # Calculate the input column of BiFPN
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c5)
        p7_x = self.p7(p6_x)

        features = [p3_x, p4_x, p5_x, p6_x, p7_x]

        # Iterate through BiFPN blocks, passing the *list* of features
        for bifpn_block in self.bifpns:
            features = bifpn_block(features)  # Pass the entire list

        return features  # Return the list of feature maps
