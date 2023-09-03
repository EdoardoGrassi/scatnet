import math

import matplotlib.pyplot as plt
import torch
import torchvision

from models.convnet import ConvNet


def plot_conv_filters(ax: plt.Axes, tensor: torch.Tensor):
    # see https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
    assert tensor is not None
    print("Tensor shape", tensor.shape)

    n, c, w, h = tensor.shape
    tensor = tensor.reshape(n * c, -1, w, h)
    print("Tensor shape", tensor.shape)

    rows = math.floor(math.sqrt(tensor.shape[0]))
    grid = torchvision.utils.make_grid(tensor, nrow=rows, normalize=True)
    print("Grid shape", grid.shape)

    ax.imshow(grid.numpy().transpose((1, 2, 0)))
    ax.axis('off')


def main():
    model = ConvNet(shape=(1, 200, 200), classes=10, classifier="mlp")
    model.load_state_dict(torch.load('results/ConvNet.pth'))

    fig, axs = plt.subplots(ncols=3)

    # plot weights of convolutional layers
    layers = [model.conv0, model.conv1, model.conv2]
    for ax, layer in zip(axs, layers):
        ax.set_title(layer._get_name())
        plot_conv_filters(ax, layer.weight)

    fig.savefig('report/images/convnet-filters.png')
    plt.show()


if __name__ == "__main__":
    main()
