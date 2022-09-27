import io
from pathlib import Path
import shutil
from typing import Callable, Final, Generator, Iterable, Optional
from urllib import request
from zipfile import ZipFile

import numpy as np
import torch
import torchvision
import unlzw3
import urllib3
from PIL import Image

from ._view_and_lumi import *



# NOTE: index starts from 1
CLASSES: Final[Generator] = range(1, 61 + 1)

SAMPLES_PER_CLASS: Final = 205

SAMPLE_AVG: Final = torch.tensor([0.45, 0.45, 0.45])
"""Pre-computed mean of the whole dataset for separate color components."""

SAMPLE_STD: Final = torch.tensor([0.35, 0.35, 0.35])
"""Pre-computed standard deviation of the whole dataset for separate color components."""

SAMPLE_AVG_GRAYSCALE: Final = 0.5
SAMPLE_STD_GRAYSCALE: Final = 0.3

def curet_meta_table():
    return CURET_VIEW_AND_LUMI


def compile_curet_dataset(src: Path, dst: Path):
    assert src.is_dir(), f"Path {src.resolve()} is not a directory"

    def decompress(encoded: Path, decoded: Path):
        fi, fo = Path(encoded), Path(decoded)
        fo.write_bytes(unlzw3.unlzw(fi.read_bytes()))

    dst.mkdir(exist_ok=True)

    meta: Final[Path] = src.joinpath("view-and-light-directions.txt")
    directions = np.loadtxt(meta, skiprows=2, usecols=(1, 2, 3, 4))
    assert directions.shape == (SAMPLES_PER_CLASS, 4)

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


def _class_to_url(index: int):
    _URL = "https://www.cs.columbia.edu/CAVE/exclude/curet/data/zips/sample{:02}.zip"
    return _URL.format(index)


def _class_to_dir(index: int):
    return Path(f"sample{index:02}")


class Curet(torchvision.datasets.DatasetFolder):
    def __init__(self,
                 root: str,
                 classes: Optional[Iterable[int]] = None,
                 download: Optional[bool] = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None:

        self.__root: Final = Path(root)
        self.__classes: Final[list[int]] = list(set(classes)) if classes is not None \
            else list(CLASSES)

        # download any missing class
        if download:
            for cls in self.__classes:
                dir = self.__root.joinpath(_class_to_dir(cls))
                if dir.exists():
                    print(f"Found folder {dir}")
                    if not len(list(dir.glob("*.Z"))) != SAMPLES_PER_CLASS:
                        pass
                else:
                    print(f"Missing folder {dir}")
                    url = _class_to_url(cls)
                    self.__retrieve_remote_data(url)

                # remove extra metadata
                if (xvpics := dir.joinpath(".xvpics/")).exists():
                    shutil.rmtree(xvpics)

        # validate class folders
        for cls in self.__classes:
            self.__validate_class_folder(cls)

        torchvision.datasets.folder.default_loader
        # NOTE: for some evil reason, torch converts extensions to lowercase
        EXTENSIONS = (".bmp.z",)
        super().__init__(root, self.__loader, EXTENSIONS, transform, target_transform)

    # override
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        class_to_idx = {f"sample{x:02}": x for x in self.__classes}
        return self.__classes, class_to_idx

    def __loader(self, path: str):
        extracted = io.BytesIO(unlzw3.unlzw(Path(path).read_bytes()))
        return Image.open(extracted).convert('RGB')

    def __retrieve_remote_data(self, url: str):
        assert url is not None

        print(f"Downloading data from {url}")
        encoded_archive_path, _ = request.urlretrieve(url)
        decoded_archive_path = self.__root

        # extract folder
        print(f"Extracting data to {decoded_archive_path}")
        with ZipFile(encoded_archive_path, "r") as archive:
            archive.extractall(path=decoded_archive_path)

        print(f"Decoding samples")
        for encoded_sample in decoded_archive_path.glob("*.bmp.Z"):
            # remove trailing .Z
            print(f"Decoding {encoded_sample}")
            decoded_sample = encoded_sample.with_suffix("")
            decoded_sample.write_bytes(
                unlzw3.unlzw(encoded_sample.read_bytes()))

    def __validate_class_folder(self, cls: int):
        if not (dir := self.__root.joinpath(_class_to_dir(cls))).is_dir():
            raise RuntimeError(f"Cannot locate {dir}")

def mean_and_std(dataset: Curet):
    assert dataset is not None

    loader: Final = torchvision.transforms.ToTensor()
    avg = torch.zeros(3)  # running average
    var = torch.zeros(3)  # running variance

    for i, (image, _) in enumerate(dataset):
        image = loader(image)
        avg += image.mean(dim=[1, 2])
        var += image.var(dim=[1, 2])

        if i % 100 == 0:
            print("Processed {}/{} ({:2}%)".format(i, len(dataset), i / len(dataset) * 100))

    avg = avg / len(dataset)
    std = torch.sqrt(var / len(dataset))
    return avg, std
