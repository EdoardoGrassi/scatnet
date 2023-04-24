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

    def __init__(self, inputs: int, outputs: int, hidden: list[int]):
        assert inputs > 0
        assert outputs > 0
        assert len(hidden) > 0 and all(x > 0 for x in hidden)

        super().__init__()

        hidden_layers = []
        for x in range(len(hidden) - 1):
            hidden_layers.append(torch.nn.Linear(in_features=hidden[x], out_features=hidden[x + 1]))
            hidden_layers.append(torch.nn.ReLU(inplace=True))

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=inputs, out_features=hidden[0]),
            torch.nn.ReLU(inplace=True),
            *hidden_layers,
            torch.nn.Linear(in_features=hidden[-1], out_features=outputs),
        )

        # self.layers = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=inputs, out_features=hidden),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(in_features=hidden, out_features=hidden),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(in_features=hidden, out_features=outputs),
        # )

    def forward(self, x):
        return self.layers(x)
