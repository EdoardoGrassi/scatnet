import re
from pathlib import Path
from typing import Callable, Optional

import torchvision
from torchvision.datasets.utils import download_and_extract_archive

class KTH_TIPS_Grey(torchvision.datasets.ImageFolder):
    """KTH Textures under varying Illumination, Pose and Scale (KTH-TIPS)

    Greyscale version of the KTH-TIPS dataset.
    See https://www.csc.kth.se/cvap/databases/kth-tips/index.html
    """

    def __init__(self,
                 root: str,
                 scales: Optional[list[int]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:

        _URL = "https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_grey_200x200.tar"
        _MD5 = "3aab2bffd539865b237cb3a63dffb14a"

        assert scales is None or all(1 <= x <= 9 for x in scales),\
            "Valid scales sit in the range [1, 9]"

        base_folder = Path(root) / "kth-tips-grey"            
        if download:
            download_and_extract_archive(_URL, download_root=str(base_folder), md5=_MD5)

        is_valid_file = None
        if scales is not None:
            # extract the scale index from the filename
            matcher = re.compile(r'scale_(\d+)')
            is_valid_file = lambda x: int(matcher.search(x).group(1)) in scales

        super().__init__(str(base_folder / "KTH_TIPS"),
                        transform=transform,
                        target_transform=target_transform,
                        is_valid_file=is_valid_file)


class KTH_TIPS_Color(torchvision.datasets.ImageFolder):
    """KTH Textures under varying Illumination, Pose and Scale (KTH-TIPS)
    
    Color version of the KTH-TIPS dataset.
    See https://www.csc.kth.se/cvap/databases/kth-tips/index.html
    """

    def __init__(self,
                 root: str,
                 scales: Optional[list[int]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:

        _URL = "https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_col_200x200.tar"
        _MD5 = "4f92fe540feb4f3c66938291e4516f6c"

        assert scales is None or all(1 <= x <= 9 for x in scales),\
            "Valid scales sit in the range [1, 9]"

        base_folder = Path(root) / "kth-tips-col"
        if download:
            download_and_extract_archive(_URL, download_root=str(base_folder), md5=_MD5)

        is_valid_file = None
        if scales is not None:
            # extract the scale index from the filename
            matcher = re.compile(r'scale_(\d+)')
            is_valid_file = lambda x: int(matcher.search(x).group(1)) in scales

        super().__init__(str(base_folder / "KTH_TIPS"),
                        transform=transform,
                        target_transform=target_transform,
                        is_valid_file=is_valid_file)