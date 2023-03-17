import torch
from kymatio.torch import Scattering2D

from models.classifiers import MLP


class ScatNet2D(torch.nn.Module):
    def __init__(self, shape: tuple[int, int], classes: int):
        assert all(x > 0 for x in shape)
        assert classes > 0

        super().__init__()
        COLOR_CHANNELS_COUNT = 1

        # TODO: justify value
        J = 2
        SCAT_M_I, SCAT_N_I = shape
        SCAT_M_O, SCAT_N_O = SCAT_M_I // (2 ** J), SCAT_N_I // (2 ** J)
        # TODO: justify value calculation
        K = 81 * COLOR_CHANNELS_COUNT  # for max_order = 2
        #inputs = K * SCAT_M_O * SCAT_N_O
        inputs = K * SCAT_M_O * SCAT_N_O
        print("Expected features shape:", inputs)

        self.features = Scattering2D(J=J, shape=shape, L=8, max_order=2)
        self.classifier = MLP(inputs=inputs, outputs=classes, hidden=1024)

    def forward(self, x: torch.Tensor):
        assert x is not None
        
        x = self.features.forward(x)
        x = x.flatten(start_dim=1) # flatten but keep batches shape
        x = self.classifier.forward(x)
        return x
