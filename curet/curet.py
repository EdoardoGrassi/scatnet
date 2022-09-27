import io
import logging
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Optional
from urllib import request
from zipfile import ZipFile

import numpy as np
import torch
import torchvision
import unlzw3
import urllib3
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset

from ._specs import *

_LOGGER = logging.getLogger(__name__)


def compile_curet_dataset(src: Path, dst: Path):
    assert src.is_dir(), f"Path {src.resolve()} is not a directory"

    def decompress(encoded: Path, decoded: Path):
        fi, fo = Path(encoded), Path(decoded)
        fo.write_bytes(unlzw3.unlzw(fi.read_bytes()))

    dst.mkdir(exist_ok=True)

    meta: Final[Path] = src.joinpath("view-and-light-directions.txt")
    directions = np.loadtxt(meta, skiprows=2, usecols=(1, 2, 3, 4))
    assert directions.shape == (CURET_SAMPLES_PER_CLASS, 4)

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


# class Curet(ImageFolder):
class Curet(Dataset):

    def __init__(
        self,
            root: Path,
            classes: Optional[Iterable[int]] = None,
            # max_view_angles: Optional[tuple[float, float]] = None,
            # max_lumi_angles: Optional[tuple[float, float]] = None,
            download: Optional[bool] = False
    ) -> None:

        assert root.is_dir(), f"Path {root.resolve()} is not a directory"
        assert classes is None or all(
            1 <= x <= CURET_NUM_CLASSES for x in classes)
        # assert all(x >= 0 for x in max_view_angles)
        # assert all(x >= 0 for x in max_lumi_angles)

        self.__root: Final[Path] = root
        self.__classes: Final[list[int]] = list(set(classes)) if classes is not None \
            else [x for x in range(1, CURET_NUM_CLASSES + 1)]

        self.__loader = torchvision.datasets.folder.default_loader

        # download any missing class
        if download:
            for cls in self.__classes:
                dir = self.__root.joinpath(_class_to_dir(cls))
                print(f"Searching folder {dir}")
                if not dir.exists() or len(list(dir.glob("*.Z"))) != CURET_SAMPLES_PER_CLASS:
                    url = _class_to_url(cls)
                    self.__retrieve_remote_data(url)
                else:
                    print(f"Found folder {dir.resolve()}")

        # validate class folders
        for cls in self.__classes:
            self.__validate_class_folder(cls)

        class_to_idx = {f"sample{x:02}": x for x in self.__classes}
        self.__samples: Final = make_dataset(
            self.__root, class_to_idx=class_to_idx, extensions=(".bmp", ".bmp.Z"))

    def __len__(self) -> int:
        return len(self.__samples)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        sample, target = self.__samples[index]

        # if self.transform is not None:
        #     sample = self.transform(sample)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return self.__loader(sample), target

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
        dir: Final = self.__root.joinpath(_class_to_dir(cls))

        if not dir.is_dir():
            raise RuntimeError(f"Cannot locate {dir}")


class Curet2(torchvision.datasets.DatasetFolder):
    def __init__(self,
                 root: str,
                 classes: list[int],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None:

        self.__classes: Final[list[int]] = set(classes)

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


def mean_and_std(dataset: Curet2):
    assert dataset is not None

    loader: Final = torchvision.transforms.ToTensor()
    avg = torch.zeros(3)  # running average
    var = torch.zeros(3)  # running variance
    for image, _ in iter(dataset):
        image = loader(image)
        avg += image.mean(dim=[1, 2])
        var += image.var(dim=[1, 2])

    avg = avg / len(dataset)
    std = torch.sqrt(var / len(dataset))
    return avg, std
