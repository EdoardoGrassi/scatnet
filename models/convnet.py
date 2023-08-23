import torch
from torch import nn

from models.classifiers import MLP


class ConvNet2D(nn.Module):
    def __init__(self, shape: tuple[int, int, int], classes: int):
        assert all(x > 0 for x in shape)
        assert classes > 0

        super().__init__()

        J = 2  # wavelet invariant scales
        L = 8  # wavelet invariant angles
        image_color_channels, SCAT_M_I, SCAT_N_I = shape
        SCAT_M_O, SCAT_N_O = SCAT_M_I // (2 ** J), SCAT_N_I // (2 ** J)

        # see https://www.kymat.io/userguide.html#output-size
        # only for order m = 2
        K = 1 + J * L + (L ** 2 * J * (J - 1)) // 2
        inputs = K * SCAT_M_O * SCAT_N_O * image_color_channels

        # based on https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4),
            nn.Tanh(),
        )
        self.classifier = MLP(inputs=120, outputs=classes, hidden=1024)

    def forward(self, x: torch.Tensor):
        assert x is not None

        x = self.features.forward(x)
        x = x.flatten(start_dim=1)  # flatten but keep batches shape
        x = self.classifier.forward(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, shape: tuple[int, int, int], classes: int, classifier: str) -> None:
        assert shape[0] in (1, 3, 4)
        assert classes > 0

        super().__init__()
        channels, w, h = shape

        self.conv0 = nn.Conv2d(channels, 128, kernel_size=(3, 3))
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.features = nn.Sequential(
            self.conv0,
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            self.conv1,
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            self.conv2,
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        match classifier:
            case "lin":
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    # TODO: try out a droput layer
                    nn.Linear(in_features=128 * 23 * 23, out_features=classes),
                )
            case "mlp":
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    # TODO: try out a droput layer
                    nn.Linear(in_features=128 * 23 * 23, out_features=128),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=128, out_features=128),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=128, out_features=classes),
                )
            case _:
                raise ValueError(f"Unknown classifier {classifier}")

    def forward(self, x: torch.Tensor):
        x = self.features.forward(x)
        x = self.classifier.forward(x)
        return x