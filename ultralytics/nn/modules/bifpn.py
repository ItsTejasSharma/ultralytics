import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import BiFPNBlock, ConvBlock, DepthwiseConvBlock  # Ensure this is correctly imported

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

        # Learnable fusion weights (Each BiFPN layer has its own set of weights)
        self.w1 = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(num_layers)])  # Top-down fusion
        self.w2 = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(num_layers)])  # Bottom-up fusion

        # Final output processing layers
        self.p3_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.p4_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.p5_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Initialize fusion weights properly
        for layer_idx in range(num_layers):
            nn.init.constant_(self.w1[layer_idx], 1.0)
            nn.init.constant_(self.w2[layer_idx], 1.0)

    def _weighted_fusion(self, features, weights):
        """Helper function for weighted feature fusion."""
        weights = F.softmax(weights, dim=0)  # Normalize weights
        weighted_sum = sum(w * f for w, f in zip(weights, features))  # Weighted sum
        return weighted_sum / (weights.sum() + self.epsilon)  # Normalize with epsilon

    def forward(self, inputs):
        p3, p4, p5 = inputs  # Extract input features
        assert p3.shape[1] == p4.shape[1] == p5.shape[1], \
            f"Input channel mismatch: {p3.shape[1]} vs {p4.shape[1]} vs {p5.shape[1]}"
        # Apply 1x1 convolutions to normalize input channels
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # Pass through BiFPN layers iteratively
        for i, bifpn_block in enumerate(self.bifpn_blocks):
            # Pass through BiFPN block (which should handle fusion inside)
            p3, p4, p5 = bifpn_block(p3, p4, p5, self.w1[i], self.w2[i])  # Pass per-layer weights

        # Apply final processing convolutions
        p3_out = self.p3_out(p3)
        p4_out = self.p4_out(p4)
        p5_out = self.p5_out(p5)

        return p5_out, p4_out, p3_out  # Reverse output order
