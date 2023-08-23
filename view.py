import itertools
from typing import Final

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datasets import KTH_TIPS_Grey

def analyze():
    a = range(10)
    b = range(10)
    cm = confusion_matrix(a, b)

    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("report/images/convnet-confmat.png")
    plt.savefig("report/images/scatnet-confmat.png")


def preview():
    transform: Final = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset: Final = KTH_TIPS_Grey(root='data/', transform=transform, download=True)

    rows, cols = 4, 4
    fig, axs = plt.subplots(nrows=rows, ncols=cols)
    for row, col in itertools.product(range(rows), range(cols)):
        index = torch.randint(len(dataset), size=(1,)).item()
        image, label = dataset[index]
        ax = axs[row][col]
        ax.set_title(f"{dataset.classes[label]}")
        ax.axis("off")
        # ax.imshow(image.permute(1, 2, 0), cmap='gray')
        ax.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')

    # fig.savefig("report/images/dataset.png")


    rows, cols = 2, 5
    fig, axs = plt.subplots(nrows=rows, ncols=cols, layout='tight')
    for cls, (row, col) in zip(dataset.classes, itertools.product(range(rows), range(cols))):
        class_idx = dataset.class_to_idx[cls]
        sample_idx = (i for i, x in enumerate(dataset.targets) if x == class_idx)
        image, _ = dataset[next(sample_idx)]

        ax = axs[row][col]
        ax.set_title(f"{cls}", fontsize=10)
        ax.axis('off')
        ax.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')

    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.savefig("report/images/dataset.png")
    plt.show()


def main():
    analyze()
    preview()

if __name__ == "__main__":
    main()
