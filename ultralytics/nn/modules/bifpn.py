import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import BiFPNBlock, ConvBlock, DepthwiseConvBlock

class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_layers = num_layers

        # Input normalization layers
        self.p3_in = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.p4_in = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.p5_in = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)

        # Additional processing convolutions
        self.p3_conv = DepthwiseConvBlock(out_channels, out_channels)
        self.p4_conv = DepthwiseConvBlock(out_channels, out_channels)
        self.p5_conv = DepthwiseConvBlock(out_channels, out_channels)

        # BiFPN blocks
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(out_channels, epsilon) 
            for _ in range(num_layers)
        ])

        # Fixed weight initialization syntax
        self.w1 = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) 
            for _ in range(num_layers)
        ])
        self.w2 = nn.ParameterList([
            nn.Parameter(torch.ones(3, dtype=torch.float32)) 
            for _ in range(num_layers)
        ])

        # Output convolutions
        self.p3_out = ConvBlock(out_channels, out_channels, 1)
        self.p4_out = ConvBlock(out_channels, out_channels, 1)
        self.p5_out = ConvBlock(out_channels, out_channels, 1)

    def forward(self, inputs):
        p3, p4, p5 = inputs
        
        # Channel normalization
        p3 = self.p3_in(p3)
        p4 = self.p4_in(p4)
        p5 = self.p5_in(p5)

        # Optional safety check
        assert p3.shape[1] == p4.shape[1] == p5.shape[1], \
            f"Channel mismatch after normalization: {p3.shape[1]} vs {p4.shape[1]} vs {p5.shape[1]}"

        # Feature processing
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # BiFPN fusion
        for i, block in enumerate(self.bifpn_blocks):
            p3, p4, p5 = block(
                p3, p4, p5,
                self.w1[i],  # Top-down weights
                self.w2[i]   # Bottom-up weights
            )

        return self.p5_out(p5), self.p4_out(p4), self.p3_out(p3)
