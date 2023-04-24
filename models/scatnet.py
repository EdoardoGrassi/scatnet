import torch
from kymatio.torch import Scattering2D

from models.classifiers import MLP


class ScatNet2D(torch.nn.Module):
    def __init__(self, shape: tuple[int, int, int], classes: int, pca: torch.Tensor):
        assert all(x > 0 for x in shape)
        assert classes > 0

        super().__init__()

        # TODO: justify values
        J = 4 # wavelet invariant scales
        L = 8 # wavelet invariant angles
        image_color_channels, w, h = shape

        # see https://www.kymat.io/userguide.html#output-size
        # only for order m = 2
        K = 1 + J * L + (L ** 2 * J * (J - 1)) // 2
        coefficients = K * (w // (2 ** J)) * (h // (2 ** J)) * image_color_channels

        self.features = Scattering2D(J=J, shape=(w, h), L=L, max_order=2)
        self.pca = pca
        self.classifier = MLP(inputs=pca.shape[-1], outputs=classes, hidden=[1024, 1024])

    def forward(self, x: torch.Tensor):
        assert x is not None
        
        x = self.features.forward(x)
        x = x.flatten(start_dim=1) # flatten but keep batches shape
        x = x.matmul(self.pca)
        x = self.classifier.forward(x)
        return x
