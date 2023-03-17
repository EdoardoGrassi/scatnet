import torch

from models.classifiers import MLP


class ConvNet2D(torch.nn.Module):
    def __init__(self, shape: tuple[int, int], classes: int):
        assert classes > 0

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(),
            torch.nn.Conv2d(),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(),
            torch.nn.Conv2d(),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(),
        )

        self.classifier = MLP(inputs=inputs, outputs=classes, hidden=1024)

    def forward(self, x):
        assert x is not None

        x = self.features.forward(x)
        x = x.flatten(start_dim=1) # flatten but keep batches shape
        x = self.classifier.forward(x)
        return x
