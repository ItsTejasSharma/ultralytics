import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import BiFPNBlock, DepthwiseConvBlock, ConvBlock  # Import required blocks

class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.num_layers = num_layers

        # Adjust input channels to match BiFPN processing size
        self.p3_conv = nn.Conv2d(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.p4_conv = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0)
        self.p5_conv = nn.Conv2d(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0)

        # BiFPN Blocks
        self.bifpn_blocks = nn.ModuleList([BiFPNBlock(out_channels, epsilon) for _ in range(num_layers)])

        # Learnable fusion weights for top-down and bottom-up pathways
        self.w1 = nn.Parameter(torch.ones(2))  # Shape (2,) for top-down fusion
        self.w2 = nn.Parameter(torch.ones(3))  # Shape (3,) for bottom-up fusion

        # Processing layers after BiFPN
        self.p3_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.p4_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.p5_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def _weighted_fusion(self, features, weights):
        """Helper function for weighted feature fusion."""
        weights = F.softmax(weights, dim=0)  # Normalize weights
        weighted_sum = sum(w * f for w, f in zip(weights, features))  # Weighted sum
        return weighted_sum / (weights.sum() + self.epsilon)  # Normalize with epsilon

    def forward(self, inputs):
        p3, p4, p5 = inputs  # Extract inputs

        # Apply 1x1 convolutions to normalize input channels
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # Pass through BiFPN layers iteratively
        for bifpn_block in self.bifpn_blocks:
            p3, p4, p5 = bifpn_block(p3, p4, p5)

        # Top-down pathway fusion
        p5_td = p5  # No changes needed for P5
        p4_td = self._weighted_fusion(
            [p4, F.interpolate(p5_td, scale_factor=2, mode="nearest")], self.w1
        )
        p3_td = self._weighted_fusion(
            [p3, F.interpolate(p4_td, scale_factor=2, mode="nearest")], self.w2[:2]
        )

        # Bottom-up pathway fusion
        p4_out = self._weighted_fusion(
            [p4, p4_td, F.interpolate(p5_td, scale_factor=0.5, mode="nearest")], self.w2
        )
        p5_out = self._weighted_fusion([p5, p5_td], self.w2[:2])

        # Apply final processing convolutions
        p3_out = self.p3_out(p3_td)
        p4_out = self.p4_out(p4_out)
        p5_out = self.p5_out(p5_out)

        return p3_out, p4_out, p5_out
