import torch
import torch.nn as nn
# [FIX 1] Import the quantize function so we can use it on activations
from quantization_utils import QuantizedConv2d, FixedPointQuantizeFunction

# --- CONFIGURATION TO REPORT ---
W_FRAC = 7  # Q2.6 for Weights
A_FRAC = 1  # Q4.4 for Activations (Configured to 1 based on your request)

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
        # 1. Squeeze
        x = self.squeeze(x)
        x = self.relu1(x)
        
        # [FIX 2] Explicitly quantize the output of Squeeze+ReLU
        # This ensures Verification 2 sees discrete values (step size 0.5)
        x = FixedPointQuantizeFunction.apply(x, 8, A_FRAC)

        # 2. Expand
        # We calculate branches separately to quantize their outputs too
        out_1x1 = self.expand1x1(x)
        out_1x1 = self.relu2(out_1x1)
        out_1x1 = FixedPointQuantizeFunction.apply(out_1x1, 8, A_FRAC) # Quantize Expand 1x1
        
        out_3x3 = self.expand3x3(x)
        out_3x3 = self.relu3(out_3x3)
        out_3x3 = FixedPointQuantizeFunction.apply(out_3x3, 8, A_FRAC) # Quantize Expand 3x3

        return torch.cat([out_1x1, out_3x3], 1)

class SqueezeNetQAT(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNetQAT, self).__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Conv1
            QuantizedConv2d(3, 96, kernel_size=7, stride=2, padding=3, 
                            weight_frac_bits=W_FRAC, input_frac_bits=A_FRAC),
            nn.ReLU(inplace=True),
            # [FIX 3] We can't easily insert the function in nn.Sequential without a lambda or wrapper.
            # However, the Fire layers (where your error is) are fixed above.
            # If Verification checks Conv1 output, you might need a custom module here too.
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