import itertools
import math
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import torchvision.transforms
from torch.utils.data import DataLoader

from curet.data import Curet, SimpleCuret


def main():
    USE_ONLY_CLASSES = list(range(1, 10 + 1))

    root: Final = Path(R"C:/data/curet/")
    root.mkdir(exist_ok=True)

    ts: Final = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(200),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    # curet: Final = Curet(root=root, classes=USE_ONLY_CLASSES, download=True, transform=ts)
    curet: Final = SimpleCuret(root=root, transform=ts)
    loader: Final = DataLoader(curet, batch_size=4, shuffle=True, num_workers=0)

    # filter by view angle
    MAX_H_ANGLE, R_ = math.radians(60), math.radians(-60)
    # MAX_V_ANGLE = math.radians(60)

    # preview some of the images
    idx_to_class = { idx: cls for cls, idx in curet.class_to_idx.items() }
    for images, labels in itertools.islice(loader, 1):
        fig, axs = plt.subplots(ncols=len(images))
        for image, label, ax in zip(images, labels, axs):
            ax.set_title(f"{idx_to_class[label.item()]}")
            ax.imshow(image.permute(1, 2, 0), cmap='gray')

        plt.show()

    print("Done")

    # mean, std = mean_and_std(curet)
    # print(f"mean = {mean}, std = {std}")

if __name__ == "__main__":
    main()
