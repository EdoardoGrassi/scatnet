import torch


class LinearSVM(torch.nn.Module):
    """
    Linear SVM classifier.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        assert in_features > 0
        assert out_features > 0

        super().__init__()
        self.layers = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron classifier.
    """

    def __init__(self, inputs: int, outputs: int, hidden: int):
        assert inputs > 0
        assert outputs > 0
        assert hidden > 0

        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=inputs, out_features=hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=hidden, out_features=hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=hidden, out_features=outputs),
        )

    def forward(self, x):
        return self.layers(x)
