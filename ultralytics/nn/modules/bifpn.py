import torch
import torch.nn as nn
from .block import BiFPNBlock, DepthwiseConvBlock, ConvBlock  # Ensure these exist

class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_layers = num_layers

        # 1. Input Normalization Layers (Critical Fix)
        self.p3_in = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.p4_in = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.p5_in = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)

        # 2. Feature Processing Convolutions
        self.p3_conv = DepthwiseConvBlock(out_channels, out_channels)
        self.p4_conv = DepthwiseConvBlock(out_channels, out_channels)
        self.p5_conv = DepthwiseConvBlock(out_channels, out_channels)

        # 3. BiFPN Blocks
        self.bifpn_blocks = nn.ModuleList([
            BiFPNBlock(out_channels, epsilon) 
            for _ in range(num_layers)
        ])

        # 4. Learnable Fusion Weights
        self.w1 = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))  # Top-down weights
            for _ in range(num_layers)
        ])
        self.w2 = nn.ParameterList([
            nn.Parameter(torch.ones(3, dtype=torch.float32))  # Bottom-up weights
            for _ in range(num_layers)
        ])

        # 5. Output Convolutions
        self.p3_out = ConvBlock(out_channels, out_channels, 1)
        self.p4_out = ConvBlock(out_channels, out_channels, 1)
        self.p5_out = ConvBlock(out_channels, out_channels, 1)

    def forward(self, inputs):
        p3, p4, p5 = inputs
        
        # Channel Normalization (Resolves Dimension Mismatch)
        p3 = self.p3_in(p3)  # Converts [128] → [out_channels]
        p4 = self.p4_in(p4)  # Converts [256] → [out_channels]
        p5 = self.p5_in(p5)  # Converts [512] → [out_channels]

        # Safety Check (Optional but Recommended)
        assert p3.shape[1] == p4.shape[1] == p5.shape[1], \
            f"Channel mismatch after normalization: {p3.shape[1]} vs {p4.shape[1]} vs {p5.shape[1]}"

        # Feature Processing
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # BiFPN Fusion
        for i, block in enumerate(self.bifpn_blocks):
            p3, p4, p5 = block(p3, p4, p5, self.w1[i], self.w2[i])

        return self.p5_out(p5), self.p4_out(p4), self.p3_out(p3)
