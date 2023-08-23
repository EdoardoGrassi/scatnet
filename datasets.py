from pathlib import Path
from typing import Callable, Optional

import torchvision
from torchvision.datasets.utils import download_and_extract_archive

class KTH_TIPS_Grey(torchvision.datasets.ImageFolder):
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

        base_folder = Path(root) / "kth-tips-grey"            
        if download:
            download_and_extract_archive(_URL, download_root=str(base_folder), md5=_MD5)

        super().__init__(str(base_folder / "KTH_TIPS"),
                         transform=transform, target_transform=target_transform)


class KTH_TIPS_Color(torchvision.datasets.ImageFolder):
    """KTH Textures under varying Illumination, Pose and Scale (KTH-TIPS) https://www.csc.kth.se/cvap/databases/kth-tips/index.html

    Color version of the KTH-TIPS dataset.
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:

        _URL = "https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_col_200x200.tar"
        _MD5 = "4f92fe540feb4f3c66938291e4516f6c"

        base_folder = Path(root) / "kth-tips-col"
        if download:
            download_and_extract_archive(_URL, download_root=str(base_folder), md5=_MD5)

        data_folder = base_folder / "KTH_TIPS"          
        super().__init__(str(data_folder),
                         transform=transform,
                         target_transform=target_transform)