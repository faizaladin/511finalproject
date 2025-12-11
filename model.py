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
        # Note: Padding=1 simulates the "zero-padding" mentioned in the paper
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
            # conv1: 96 filters, 7x7, stride 2
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3), # padding=3 maintains size logic before stride
            nn.ReLU(inplace=True),
            
            # maxpool1: 3x3, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire2: s16, e64, e64 -> 128
            Fire(96, 16, 64, 64),
            # fire3: s16, e64, e64 -> 128
            Fire(128, 16, 64, 64),
            # fire4: s32, e128, e128 -> 256
            Fire(128, 32, 128, 128),
            
            # maxpool4: 3x3, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire5: s32, e128, e128 -> 256
            Fire(256, 32, 128, 128),
            # fire6: s48, e192, e192 -> 384
            Fire(256, 48, 192, 192),
            # fire7: s48, e192, e192 -> 384
            Fire(384, 48, 192, 192),
            # fire8: s64, e256, e256 -> 512
            Fire(384, 64, 256, 256),
            
            # maxpool8: 3x3, stride 2 (This matches your table)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # fire9: s64, e256, e256 -> 512
            Fire(512, 64, 256, 256),
        )

        # Classifier
        self.classifier = nn.Sequential(
            # Dropout 50%
            nn.Dropout(p=0.5),
            
            # conv10: 1x1, stride 1. 
            # ADAPTATION: The table says "1000" (ImageNet), but we use num_classes (10) for CIFAR.
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            
            # NOTE: Your table lists "conv10" then "avgpool10". 
            # It does NOT list a ReLU between them. We strictly follow the table here.
            
            # avgpool10: Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Weight Initialization (Crucial for training from scratch)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier[1]:
                    # Special init for the classifier layer (suggested by paper authors later)
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)