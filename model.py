import torch
import torch.nn as nn
import torch.nn.init as init

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        
        # Squeeze Layer
        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Expand Layer (1x1 filters)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Expand Layer (3x3 filters)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.squeeze(x))
        return torch.cat([
            self.relu2(self.expand1x1(x)),
            self.relu3(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3), 
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire2
            Fire(96, 16, 64, 64),
            # fire3
            Fire(128, 16, 64, 64),
            # fire4
            Fire(128, 32, 128, 128),
            
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire5
            Fire(256, 32, 128, 128),
            # fire6
            Fire(256, 48, 192, 192),
            # fire7
            Fire(384, 48, 192, 192),
            # fire8
            Fire(384, 64, 256, 256),
            
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire9
            Fire(512, 64, 256, 256),
        )

        # Classifier
        self.classifier = nn.Sequential(
            # Dropout 50%
            nn.Dropout(p=0.5),
            
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier[1]:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)