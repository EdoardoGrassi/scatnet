import itertools
import ssl
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import DTD, Food101

ssl._create_default_https_context = ssl._create_unverified_context


def main():
    transform: Final = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop(200),
        transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset: Final = DTD(root='data/', transform=transform, download=True)
    print(dataset.extra_repr())
    print("Classes:", dataset.classes)

    # preview some of the images
    idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    # for images, labels in dataset:
    #     fig, axs = plt.subplots(ncols=len(images))
    #     for image, label, ax in zip(images, labels, axs):
    #         ax.set_title(f"{idx_to_class[label.item()]}")
    #         #ax.imshow(image.permute(1, 2, 0), cmap='gray')
    #         ax.imshow(image.permute(1, 2, 0))

    #     plt.show()

    rows, cols = 4, 4
    fig, axs = plt.subplots(nrows=rows, ncols=cols)
    for row, col in itertools.product(range(rows), range(cols)):
        index = torch.randint(len(dataset), size=(1,)).item()
        image, label = dataset[index]
        ax = axs[row][col]
        ax.set_title(f"{idx_to_class[label]}")
        ax.axis("off")
        # ax.imshow(image.permute(1, 2, 0), cmap='gray')
        ax.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')

    fig.show()
    plt.show()

    # mean, std = mean_and_std(curet)
    # print(f"mean = {mean}, std = {std}")


if __name__ == "__main__":
    main()
