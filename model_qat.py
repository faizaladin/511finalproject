import torch
import torch.nn as nn
# [FIX 1] Import FixedPointQuantizeFunction
from quantization_utils import QuantizedConv2d, FixedPointQuantizeFunction

# --- CONFIGURATION TO REPORT ---
W_FRAC = 7  # Q2.6 for Weights
A_FRAC = 1  # Q4.4 for Activations

class FireQAT(nn.Module):
    def __init__(self, in_channels, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireQAT, self).__init__()
        
        # Squeeze
        self.squeeze = QuantizedConv2d(in_channels, squeeze_planes, kernel_size=1, 
                                       weight_frac_bits=W_FRAC, input_frac_bits=A_FRAC)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Expand 1x1
        self.expand1x1 = QuantizedConv2d(squeeze_planes, expand1x1_planes, kernel_size=1,
                                         weight_frac_bits=W_FRAC, input_frac_bits=A_FRAC)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Expand 3x3
        self.expand3x3 = QuantizedConv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1,
                                         weight_frac_bits=W_FRAC, input_frac_bits=A_FRAC)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Squeeze Path
        x = self.squeeze(x)
        x = self.relu1(x)
        
        # [FIX 2] Explicitly quantize here!
        # This converts the Float output of ReLU into the Discrete steps your test expects.
        x = FixedPointQuantizeFunction.apply(x, 8, A_FRAC)

        # 2. Expand Paths
        # We also quantize these outputs to be consistent
        out1x1 = self.expand1x1(x)
        out1x1 = self.relu2(out1x1)
        out1x1 = FixedPointQuantizeFunction.apply(out1x1, 8, A_FRAC)

        out3x3 = self.expand3x3(x)
        out3x3 = self.relu3(out3x3)
        out3x3 = FixedPointQuantizeFunction.apply(out3x3, 8, A_FRAC)

        return torch.cat([out1x1, out3x3], 1)

class SqueezeNetQAT(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNetQAT, self).__init__()
        
        self.features = nn.Sequential(
            QuantizedConv2d(3, 96, kernel_size=7, stride=2, padding=3, 
                            weight_frac_bits=W_FRAC, input_frac_bits=A_FRAC),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            FireQAT(96, 16, 64, 64),
            FireQAT(128, 16, 64, 64),
            FireQAT(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            FireQAT(256, 32, 128, 128),
            FireQAT(256, 48, 192, 192),
            FireQAT(384, 48, 192, 192),
            FireQAT(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            FireQAT(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            QuantizedConv2d(512, num_classes, kernel_size=1, 
                            weight_frac_bits=W_FRAC, input_frac_bits=A_FRAC),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)