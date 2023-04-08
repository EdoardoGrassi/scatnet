import torch
import torchvision
import matplotlib.pyplot as plt
from models.convnet import ConvNet2D

from torchvision.datasets import Food101

from pathlib import Path
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
    dataset = Food101(root='data', download=True)
    test_img_data, _ = dataset[0]
    model = ConvNet2D(shape=test_img_data.shape, classes=len(dataset.classes))


    #with open([x for x in Path('checkpoints/').iterdir()][0]) as f:

    model: ConvNet2D = torch.load('checkpoints/checkpoint_4690.pt')

    # plot weights of convolutional layers
    layers = [0, 3, 6]
    for name, module in model.features.named_modules():
        if name in layers:
            conv = cast(torch.nn.Conv2d, module)
            plot(conv.weight)


if __name__ == "__main__":
    main()
