import torch
from torch import nn

from models.classifiers import MLP


class ConvNet2D(nn.Module):
    def __init__(self, shape: tuple[int, int, int], classes: int):
        assert all(x > 0 for x in shape)
        assert classes > 0

        super().__init__()

        J = 2 # wavelet invariant scales
        L = 8 # wavelet invariant angles
        image_color_channels, SCAT_M_I, SCAT_N_I = shape
        SCAT_M_O, SCAT_N_O = SCAT_M_I // (2 ** J), SCAT_N_I // (2 ** J)

        # see https://www.kymat.io/userguide.html#output-size
        # only for order m = 2
        K = 1 + J * L + (L ** 2 * J * (J - 1)) // 2
        inputs = K * SCAT_M_O * SCAT_N_O * image_color_channels

        # based on https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
            nn.Tanh(),
        )
        self.classifier = MLP(inputs=120, outputs=classes, hidden=1024)

    def forward(self, x: torch.Tensor):
        assert x is not None

        x = self.features.forward(x)
        x = x.flatten(start_dim=1) # flatten but keep batches shape
        x = self.classifier.forward(x)
        return x



class FoodConvNet(ConvNet2D):
    def __init__(self):
        super().__init__(shape=(), classes=10)