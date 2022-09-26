from typing import Final
from pathlib import Path
import unlzw3

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

_CURET_SAMPLES_PER_CLASS: Final = 201

# NOTE: index starts from 1
_CURET_INDEX_TO_LABELS: Final[dict[int, str]] = {
    1: "Felt",
    2: "Polyester",
    3: "Terrycloth",
    4: "Rough Plastic",
    5: "Leather",
    6: "Sandpaper",
    7: "Velvet",
    8: "Pebbles",
    9: "Frosted Glass",
    10: "Plaster_a",
    11: "Plaster_b",
    12: "Rough Paper",
    13: "Artificial Grass",
    14: "Roof Shingle",
    15: "Aluminum Foil",
    16: "Cork",
    17: "Rough Tile",
    18: "Rug_a",
    19: "Rug_b",
    20: "Styrofoam",
    21: "Sponge",
    22: "Lambswool",
    23: "Lettuce Leaf",
    24: "Rabbit Fur",
    25: "Quarry Tile",
    26: "Loofa",
    27: "Insulation",
    28: "Crumpled Paper",
    29: "(2 zoomed)",
    30: "(11 zoomed)",
    31: "(12 zoomed)",
    32: "(14 zoomed)",
    33: "Slate_a",
    34: "Slate_b",
    35: "Painted Spheres",
    36: "Limestone",
    37: "Brick_a",
    38: "Ribbed Paper",
    39: "Human Skin",
    40: "Straw",
    41: "Brick_b",
    42: "Corduroy",
    43: "Salt Crystals",
    44: "Linen",
    45: "Concrete_a",
    46: "Cotton",
    47: "Stones",
    48: "Brown Bread",
    49: "Concrete_b",
    50: "Concrete_c",
    51: "Corn Husk",
    52: "White Bread",
    53: "Soleirolia Plant",
    54: "Wood_a",
    55: "Orange Peel",
    56: "Wood_b",
    57: "Peacock Feather",
    58: "Tree Bark",
    59: "Cracker_a",
    60: "Cracker_b",
    61: "Moss"
}


def compile_curet_dataset(src: Path, dst: Path):
    assert src.is_dir(), f"Path {src.resolve()} is not a directory"

    def decompress(encoded: Path, decoded: Path):
        fi, fo = Path(encoded), Path(decoded)
        fo.write_bytes(unlzw3.unlzw(fi.read_bytes()))

    dst.mkdir(exist_ok=True)

    meta: Final[Path] = src.joinpath("view-and-light-directions.txt")
    directions = np.loadtxt(meta, skiprows=2, usecols=(1, 2, 3, 4))
    assert directions.shape == (_CURET_SAMPLES_PER_CLASS, 4)

    for dir in src.glob("sample*/"):
        sampledir = dst.joinpath(dir.name)
        sampledir.mkdir(exist_ok=True)

        for file in dir.glob("*.bmp.Z"):
            encoded = file
            decoded = sampledir.joinpath(file.stem).with_suffix(".bmp")

            if not decoded.exists():
                print(f"Extracting {file}")
                decompress(encoded, decoded)
            else:
                print(f"Skipping {file}")


def preview_curet_dataset(root: Path):
    assert root.is_dir(), f"Path {root.resolve()} is not a directory"

    curet = Curet(root=root)
    loader = DataLoader(curet, batch_size=4, shuffle=True, num_workers=4)

    # preview some of the images
    images, labels = next(iter(loader))
    fig, axs = plt.subplots(ncols=len(images))
    for image, label, ax in zip(images, labels, axs):
        # parse "sampleXX" string to extract index
        class_index = int(curet.classes[label.item()][-2:])
        ax.set_title(_CURET_INDEX_TO_LABELS[class_index])
        ax.imshow(image.squeeze(), cmap='gray')  # , vmin=0, vmax=255)

    fig.show()
    plt.show()
    print("Done")


class Curet(ImageFolder):
    def __init__(self, root: Path, transform) -> None:
        assert root.is_dir(), f"Path {root.resolve()} is not a directory"

        # transform = transforms.Compose([
        #     transforms.CenterCrop(200),
        #     transforms.Resize((32, 32)),
        #     transforms.Grayscale(),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize()
        # ])
        super().__init__(root=str(root), transform=transform)


if __name__ == "__main__":
    src: Final = Path(R"C:\data\curet\raw")
    dst: Final = Path(R"C:\data\curet\data")

    compile_curet_dataset(src, dst)
    preview_curet_dataset(dst)
