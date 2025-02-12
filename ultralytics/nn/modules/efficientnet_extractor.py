import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Extractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        if self.pretrained:
          self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
          self.model = efficientnet_b0(weights=None)

        self.p3 = None
        self.p4 = None

        # Hook the layers where we want to extract features
        #  Using named children for robustness
        for name, module in self.model.features.named_children():
            if name == '2':  # P3/8
                module.register_forward_hook(self.get_p3)
            elif name == '4':  # P4/16
                module.register_forward_hook(self.get_p4)

    def get_p3(self, module, input, output):
        self.p3 = output

    def get_p4(self, module, input, output):
        self.p4 = output

    def forward(self, x):
        # Run through the EfficientNet features
        out = self.model.features(x)
        # Capture P5 (before avgpool and classifier)
        p5 = out
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.model.classifier(out) #we need this part

        return [self.p3, self.p4, p5]
