from typing import Literal
import torch
import torch.nn as nn


class ScatNet2D(nn.Module):
    """
    Simple CNN with 3x3 convs based on VGG
    """

    def __init__(self, in_channels: int, out_channels: int, classifier: Literal['cnn', 'mlp', 'lin']='cnn'):
        assert in_channels > 0
        assert out_channels > 0

        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__classifier = classifier
        self.build()

    def build(self):
        cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
        layers = []
        self.K = self.__in_channels
        self.bn = nn.BatchNorm2d(self.K)
        if self.__classifier == 'cnn':
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(self.__in_channels, v,
                                       kernel_size=3, padding=1)
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    self.__in_channels = v

            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(1024*4, self.__out_channels)

        elif self.__classifier == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.K * 8 * 8, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, self.__out_channels))
            self.features = None

        elif self.__classifier == 'lin':
            self.classifier = nn.Linear(self.K * 8 * 8, self.__out_channels)
            self.features = None

    def forward(self, x: torch.Tensor):
        x = self.bn(x.view(-1, self.K, 8, 8))
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LinearSVM(torch.nn.Module):
    
    def __init__(self, in_features: int, out_features: int) -> None:
        assert all(i > 0 for i in (in_features, out_features))

        super().__init__()
        self.__classifier = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        return self.__classifier(x)