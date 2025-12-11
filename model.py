import torch
import torch.nn as nn
import torch.nn.init as init

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNetCIFAR, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # fire2
            Fire(96, 16, 64, 64),
            # fire3
            Fire(128, 16, 64, 64),
            # fire4
            Fire(128, 32, 128, 128),
            
            # maxpool/2 (Strategy 3: Late pooling)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire5
            Fire(256, 32, 128, 128),
            # fire6
            Fire(256, 48, 192, 192),
            # fire7
            Fire(384, 48, 192, 192),
            # fire8
            Fire(384, 64, 256, 256),
            
            # maxpool/2
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire9
            Fire(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            
            # --- MODIFIED FOR CIFAR-10 (10 Classes) ---
            # Output channels changed from 1000 to 10
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

# Initialize for CIFAR-10
model = SqueezeNetCIFAR(num_classes=10)

# Verify input/output shape
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")