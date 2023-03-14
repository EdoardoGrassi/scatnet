import torch
import torchvision
import matplotlib.pyplot as plt
from models.convnet import ConvNet2D

from typing import cast


def plot(tensor: torch.Tensor):
    assert tensor is not None

    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True)
    fig = plt.figure(figsize=(nrow, rows))
    fig.imshow(grid.numpy().transpose((1, 2, 0)))


def main():
    FORMAT = ()
    LABELS = 10
    model = ConvNet2D(shape=FORMAT, classes=LABELS)

    with open('.checkpoints/') as f:
        model: ConvNet2D = torch.load(f)

    # plot weights of convolutional layers
    layers = [0, 3, 6]
    for name, module in model.layers.named_modules():
        if name in layers:
            conv = cast(torch.nn.Conv2d, module)
            plot(conv.weight)


if __name__ == "__main__":
    main()
