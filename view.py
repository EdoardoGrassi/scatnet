from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from curet.curet import Curet, Curet2, mean_and_std

if __name__ == "__main__":
    CLASSES = list(range(1, 10 + 1))

    root: Final = Path(R"C:/data/curet")
    root.mkdir(exist_ok=True)

    curet: Final = Curet2(root=root, classes=CLASSES) #, download=True)
    print("Loaded")
    loader: Final = DataLoader(curet, batch_size=4, shuffle=True, num_workers=4)

    # preview some of the images
    # images, labels = next(iter(loader))
    # fig, axs = plt.subplots(ncols=len(images))
    # for image, label, ax in zip(images, labels, axs):
    #     # parse "sampleXX" string to extract index
    #     # class_index = int(curet.classes[label.item()][-2:])
    #     # ax.set_title(CURET_INDEX_TO_LABELS[class_index])
    #     ax.imshow(image.squeeze(), cmap='gray')  # , vmin=0, vmax=255)

    # fig.show()
    # plt.show()
    # print("Done")

    mean, std = mean_and_std(curet)
    print(f"mean = {mean}, std = {std}")
