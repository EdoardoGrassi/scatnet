import torch
from kymatio.torch import Scattering2D

from models.classifiers import MLP


class ScatNet2D(torch.nn.Module):
    def __init__(self, shape: tuple[int, int, int], classes: int):
        assert all(x > 0 for x in shape)
        assert classes > 0

        super().__init__()

        # TODO: justify values
        J = 3 # wavelet invariant scales
        L = 8 # wavelet invariant angles
        image_color_channels, SCAT_M_I, SCAT_N_I = shape
        SCAT_M_O, SCAT_N_O = SCAT_M_I // (2 ** J), SCAT_N_I // (2 ** J)

        # see https://www.kymat.io/userguide.html#output-size
        # only for order m = 2
        K = 1 + J * L + (L ** 2 * J * (J - 1)) // 2
        inputs = K * SCAT_M_O * SCAT_N_O * image_color_channels

        self.features = Scattering2D(J=J, shape=(SCAT_M_I, SCAT_N_I), L=L, max_order=2)
        self.classifier = MLP(inputs=inputs, outputs=classes, hidden=1024)

    def forward(self, x: torch.Tensor):
        assert x is not None
        
        x = self.features.forward(x)
        x = x.flatten(start_dim=1) # flatten but keep batches shape
        x = self.classifier.forward(x)
        return x
