import hashlib
from pathlib import Path
from typing import Callable, Optional

import torchvision
from torchvision.datasets.utils import download_and_extract_archive

class KTHTIPS(torchvision.datasets.ImageFolder):
    """KTH Textures under varying Illumination, Pose and Scale (KTH-TIPS) https://www.csc.kth.se/cvap/databases/kth-tips/index.html

    Greyscale version of the KTH-TIPS dataset.
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:

        _URL = "https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_grey_200x200.tar"
        _MD5 = "3aab2bffd539865b237cb3a63dffb14a"

        base_folder = Path(root) / "kth"            
        if download:
            download_and_extract_archive(_URL, download_root=str(base_folder), md5=_MD5)

        super().__init__(str(base_folder / "KTH_TIPS"),
                         transform=transform, target_transform=target_transform)


class KTHTIPSColor(torchvision.datasets.ImageFolder):
    pass